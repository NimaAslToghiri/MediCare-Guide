# DoctorAssistant/medassist/views.py

import json
import logging
import os

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .agents.medical_assistant_graph import compiled_medical_assistant_graph, AgentState
from .models import MedicalDocument

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_or_initialize_graph_state(request) -> AgentState:
    logger.debug("VIEW: get_or_initialize_graph_state called")
    session_key = 'langgraph_medical_assistant_state_v2'  # Changed key to ensure fresh state with new structure

    if session_key not in request.session:
        logger.info(f"VIEW: Session state for key '{session_key}' not found. Initializing.")
        initial_state_data: AgentState = {
            "user_id": request.user.id,
            "chat_history": [],
            "current_user_input_text": None,
            "medical_document_id": None,
            "ocr_extracted_text": None,
            "ocr_error_message": None,
            "final_llm_response": None,
        }
        request.session[session_key] = initial_state_data
        logger.info("VIEW: New LangGraph session state initialized.")
    else:
        logger.debug(f"VIEW: Session state for key '{session_key}' found. Loading existing state.")
        current_state = request.session[session_key]
        # Ensure all keys are present, add if missing (for robustness if AgentState changes)
        default_keys: AgentState = {
            "user_id": request.user.id, "chat_history": [], "current_user_input_text": None,
            "medical_document_id": None, "ocr_extracted_text": None,
            "ocr_error_message": None, "final_llm_response": None
        }
        updated = False
        for key, default_value in default_keys.items():
            if key not in current_state:
                logger.warning(f"VIEW: Key '{key}' missing in session state, adding with default.")
                current_state[key] = default_value
                updated = True
        if updated:
            request.session[session_key] = current_state

    loaded_state = request.session[session_key]
    # Ensure chat_history is a list, defensively
    if not isinstance(loaded_state.get("chat_history"), list):
        logger.warning("VIEW: chat_history in session was not a list. Resetting to empty list.")
        loaded_state["chat_history"] = []
        request.session[session_key] = loaded_state  # Save correction

    logger.debug(
        f"VIEW: State loaded/initialized: user_id={loaded_state.get('user_id')}, chat_history_len={len(loaded_state.get('chat_history', []))}")
    return loaded_state


@login_required
def chat_view(request):
    logger.debug("VIEW: chat_view called")
    current_graph_session_state = get_or_initialize_graph_state(request)
    chat_messages_for_template = current_graph_session_state.get("chat_history", [])
    logger.debug(f"VIEW: Rendering chat_view with {len(chat_messages_for_template)} messages.")
    return render(request, 'medassist/chat.html', {'chat_messages_json': json.dumps(chat_messages_for_template)})


@csrf_exempt
@login_required
def chat_api_endpoint(request):
    logger.debug("VIEW: chat_api_endpoint called (Method: %s)", request.method)
    if request.method != 'POST':
        logger.warning("VIEW: Non-POST request to chat_api_endpoint.")
        return JsonResponse({"status": "error", "message": "Only POST requests are allowed."}, status=405)

    if not compiled_medical_assistant_graph:
        logger.error("VIEW: CRITICAL - compiled_medical_assistant_graph is None.")
        return JsonResponse({"status": "error", "message": "AI Assistant is currently unavailable (config error)."},
                            status=503)

    user = request.user
    current_session_state = get_or_initialize_graph_state(request)
    input_text_from_user = request.POST.get('text_input', '').strip()
    uploaded_file_from_user = request.FILES.get('uploaded_file')

    logger.info(
        f"VIEW: Received input: text='{input_text_from_user[:50]}...', file_uploaded={'Yes' if uploaded_file_from_user else 'No'}")

    run_input_state: AgentState = {
        "user_id": user.id,
        "chat_history": list(current_session_state.get("chat_history", [])),
        "current_user_input_text": input_text_from_user if input_text_from_user else None,
        "medical_document_id": None,
        "ocr_extracted_text": None, "ocr_error_message": None, "final_llm_response": None
    }
    user_message_for_history = ""

    if uploaded_file_from_user:
        original_file_name = uploaded_file_from_user.name
        logger.info(f"VIEW: Processing uploaded file: {original_file_name}")
        file_extension = os.path.splitext(original_file_name)[1].lower()
        doc_model_type = 'TEXT'
        if file_extension == ".pdf":
            doc_model_type = 'PDF'
        elif file_extension in [".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".gif"]:
            doc_model_type = 'IMAGE'
        else:
            logger.warning(f"VIEW: File extension {file_extension} being saved as generic type.")

        try:
            document_instance = MedicalDocument(user=user, document_type=doc_model_type, file=uploaded_file_from_user)
            document_instance.save()
            run_input_state["medical_document_id"] = document_instance.id
            logger.info(f"VIEW: Saved file. MedicalDocument ID: {document_instance.id}, Type: {doc_model_type}")
            user_message_for_history = f"Uploaded file: {original_file_name}"
        except Exception as e:
            logger.exception(f"VIEW: Error saving file '{original_file_name}': {e}")
            run_input_state["chat_history"].append({"sender": "user",
                                                    "message": input_text_from_user if input_text_from_user else f"Attempted to upload {original_file_name}"})
            run_input_state["chat_history"].append(
                {"sender": "ai", "message": f"Sorry, error saving your file '{original_file_name}'."})
            request.session['langgraph_medical_assistant_state_v2'] = run_input_state
            request.session.modified = True
            return JsonResponse({"status": "error_file_save", "message": f"Error saving file.",
                                 "chat_history": run_input_state["chat_history"]})

    if input_text_from_user:
        user_message_for_history = f"{user_message_for_history}\nUser message: \"{input_text_from_user}\"" if user_message_for_history else input_text_from_user

    if not user_message_for_history and not run_input_state["medical_document_id"]:
        logger.warning("VIEW: No text input and no file. Returning 400.")
        return JsonResponse({"status": "error", "message": "Please type a message or upload a file."}, status=400)

    if user_message_for_history:  # Add consolidated user message
        run_input_state["chat_history"].append({"sender": "user", "message": user_message_for_history.strip()})

    logger.debug(
        f"VIEW: Prepared run_input_state for LangGraph: { {k: v for k, v in run_input_state.items() if k != 'chat_history'} } history_len={len(run_input_state['chat_history'])}")

    final_graph_output_state = None
    ai_response_message = "Error: AI assistant did not provide a response."

    try:
        logger.info("VIEW: Invoking LangGraph stream...")
        for step_output in compiled_medical_assistant_graph.stream(run_input_state):
            logger.debug(f"VIEW: Graph stream step: { {k: v for k, v in step_output.items() if k != 'chat_history'} }")
            if "text_responder_node" in step_output:  # This is the last node before END
                final_graph_output_state = step_output["text_responder_node"]

        if final_graph_output_state and final_graph_output_state.get("final_llm_response"):
            ai_response_message = final_graph_output_state["final_llm_response"]
            logger.info(f"VIEW: Got AI response from graph: '{ai_response_message[:100]}...'")
        elif final_graph_output_state:
            logger.warning("VIEW: 'final_llm_response' missing from text_responder_node output.")
            ai_response_message = "I had trouble formulating a response. Please try again."
        else:
            logger.error("VIEW: LangGraph stream finished, but final state for text_responder_node not captured.")
            ai_response_message = "An unexpected issue occurred with the AI assistant processing."

    except Exception as e:
        logger.exception(f"VIEW: CRITICAL ERROR during LangGraph invocation: {e}")
        ai_response_message = "Sorry, I encountered a critical error. Please try again later."
        final_graph_output_state = run_input_state  # Fallback to preserve user history

    if final_graph_output_state is None: final_graph_output_state = run_input_state
    if not isinstance(final_graph_output_state.get("chat_history"), list):
        final_graph_output_state["chat_history"] = list(run_input_state.get("chat_history", []))

    final_graph_output_state["chat_history"].append({"sender": "ai", "message": ai_response_message})

    # Save OCR text to DB if successful
    if final_graph_output_state.get("medical_document_id") and \
            final_graph_output_state.get("ocr_extracted_text") and \
            not final_graph_output_state.get("ocr_error_message"):
        try:
            doc_to_update = MedicalDocument.objects.get(id=final_graph_output_state["medical_document_id"])
            doc_to_update.raw_text = final_graph_output_state["ocr_extracted_text"]
            doc_to_update.save(update_fields=['raw_text'])
            logger.info(f"VIEW: Saved OCR text to MedicalDocument ID {doc_to_update.id}")
        except Exception as e:
            logger.exception(f"VIEW: Error saving OCR text to DB: {e}")

    request.session['langgraph_medical_assistant_state_v2'] = final_graph_output_state
    request.session.modified = True
    logger.info("VIEW: Updated session state saved.")

    logger.debug("--- VIEW: Exiting chat_api_endpoint (SUCCESS) ---")
    return JsonResponse({
        "status": "success", "message": ai_response_message,
        "chat_history": final_graph_output_state["chat_history"]
    })
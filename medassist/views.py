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
    session_key = 'langgraph_medical_assistant_state_v3'  # Changed key for new state structure

    if session_key not in request.session:
        logger.info(f"VIEW: Session state for key '{session_key}' not found. Initializing.")
        initial_state_data: AgentState = {
            "user_id": request.user.id, "chat_history": [], "current_user_input_text": None,
            "medical_document_id": None, "ocr_extracted_text": None,
            "ocr_error_message": None,
            "query_for_rag": None,  # New RAG field
            "translated_query_for_rag": None,  # New RAG field
            "retrieved_rag_documents": [],  # New RAG field (init as empty list)
            "initial_reasoning_output": None,
            "reasoning_error_message": None, "final_llm_response": None,
            "revisor_error_message": None
        }
        request.session[session_key] = initial_state_data
        logger.info("VIEW: New LangGraph session state initialized (with RAG fields).")
    else:
        # ... (your existing logic to ensure all keys are present) ...
        # MAKE SURE THE default_keys HERE MATCHES THE AgentState definition above
        current_state = request.session[session_key]
        default_keys: AgentState = {
            "user_id": request.user.id, "chat_history": [], "current_user_input_text": None,
            "medical_document_id": None, "ocr_extracted_text": None,
            "ocr_error_message": None, "query_for_rag": None, "translated_query_for_rag": None,
            "retrieved_rag_documents": [], "initial_reasoning_output": None,
            "reasoning_error_message": None, "final_llm_response": None,
            "revisor_error_message": None
        }
        updated = False
        for key, default_value in default_keys.items():
            if key not in current_state:
                current_state[key] = default_value
                updated = True
        if updated:
            request.session[session_key] = current_state

    loaded_state = request.session[session_key]
    if not isinstance(loaded_state.get("chat_history"), list):
        loaded_state["chat_history"] = []
    if not isinstance(loaded_state.get("retrieved_rag_documents"), list):  # Ensure RAG docs is a list
        loaded_state["retrieved_rag_documents"] = []
    request.session[session_key] = loaded_state  # Save corrections
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

    # This `run_input_state` is prepared for the current LangGraph execution.
    # It starts with the chat_history from the *previous* turn.
    run_input_state: AgentState = {
        "user_id": user.id,
        "chat_history": list(current_session_state.get("chat_history", [])),  # History up to the previous turn
        "current_user_input_text": input_text_from_user if input_text_from_user else None,
        "medical_document_id": None,  # Will be set if file is processed
        "ocr_extracted_text": None,
        "ocr_error_message": None,
        "initial_reasoning_output": None,  # Ensure all AgentState fields are present
        "reasoning_error_message": None,
        "final_llm_response": None,
        "revisor_error_message": None
    }
    user_message_for_history = ""  # This will be the content of the user's current message

    # --- File Handling ---
    if uploaded_file_from_user:
        original_file_name = uploaded_file_from_user.name
        logger.info(f"VIEW: Processing uploaded file: {original_file_name}")
        file_extension = os.path.splitext(original_file_name)[1].lower()
        doc_model_type = 'TEXT'  # Default
        if file_extension == ".pdf":
            doc_model_type = 'PDF'
        elif file_extension in [".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".gif"]:  # Common image types
            doc_model_type = 'IMAGE'
        else:
            logger.warning(
                f"VIEW: File extension {file_extension} for '{original_file_name}' being saved as generic type, OCR might not support.")

        try:
            document_instance = MedicalDocument(user=user, document_type=doc_model_type, file=uploaded_file_from_user)
            document_instance.save()
            run_input_state["medical_document_id"] = document_instance.id
            logger.info(f"VIEW: Saved file. MedicalDocument ID: {document_instance.id}, Type: {doc_model_type}")
            user_message_for_history = f"Uploaded file: {original_file_name}"  # Part 1 of current user message content
        except Exception as e:
            logger.exception(f"VIEW: Error saving file '{original_file_name}': {e}")
            # If file save fails, add user's attempt and an error to history and return
            temp_chat_history = list(run_input_state.get("chat_history", []))
            temp_chat_history.append({"sender": "user",
                                      "message": input_text_from_user if input_text_from_user else f"Attempted to upload {original_file_name}"})
            temp_chat_history.append(
                {"sender": "ai",
                 "message": f"Sorry, there was an error saving your file '{original_file_name}'. Please try again."})

            # Update session state with this intermediate history before returning
            current_session_state["chat_history"] = temp_chat_history
            request.session['langgraph_medical_assistant_state_v2'] = current_session_state
            request.session.modified = True
            return JsonResponse({"status": "error_file_save", "message": f"Error saving file.",
                                 "chat_history": temp_chat_history})

    # --- Consolidate User's Current Message for History ---
    if input_text_from_user:
        if user_message_for_history:  # A file was also uploaded
            user_message_for_history += f"\nUser message: \"{input_text_from_user}\""  # Append text to file upload message
        else:  # Only text was input
            user_message_for_history = input_text_from_user

    # Check if there's any actual input for this turn
    if not user_message_for_history and not run_input_state["medical_document_id"]:
        logger.warning(
            "VIEW: No text input and no file successfully processed to create a user message. Returning 400.")
        # It's debatable whether to return an error or let the agent handle "empty" input.
        # For now, returning an error as no new action for the agent.
        return JsonResponse({"status": "error", "message": "Please type a message or upload a file."}, status=400)

    # **POINT 1: Add current user's consolidated message to the chat_history that goes INTO the graph**
    if user_message_for_history:
        run_input_state["chat_history"].append({"sender": "user", "message": user_message_for_history.strip()})
        logger.debug(
            f"VIEW: Appended current user message to run_input_state.chat_history. New length: {len(run_input_state['chat_history'])}")

    logger.debug(
        f"VIEW: Prepared run_input_state for LangGraph: { {k: v for k, v in run_input_state.items() if k != 'chat_history'} } history_len={len(run_input_state['chat_history'])}")

    # --- LangGraph Invocation ---
    final_graph_output_state = None  # This will hold the complete state *after* the graph run
    ai_response_message = "Error: AI assistant did not provide a response."  # Default error

    try:
        logger.info("VIEW: Invoking LangGraph stream...")
        # The stream yields the state updates. The last one for the relevant node contains its full state.
        for step_output in compiled_medical_assistant_graph.stream(run_input_state):
            logger.debug(f"VIEW: Graph stream step output keys: {list(step_output.keys())}")
            # `step_output` is a dictionary like {'node_name': AgentState_after_node_run}
            # We are interested in the state after the 'revisor_node' (our last agent) has run.
            if "revisor_node" in step_output:
                final_graph_output_state = step_output["revisor_node"]  # Capture the state from the revisor
                logger.debug("VIEW: Captured state from 'revisor_node' output.")

        if final_graph_output_state and final_graph_output_state.get("final_llm_response"):
            ai_response_message = final_graph_output_state["final_llm_response"]
            logger.info(f"VIEW: Got AI response from graph: '{ai_response_message[:100]}...'")
        elif final_graph_output_state:  # Revisor node ran, but no specific response message
            logger.warning("VIEW: 'final_llm_response' missing or None in revisor_node's output state.")
            ai_response_message = "I've processed your input, but I'm having a bit of trouble formulating a full response right now. Could you try rephrasing or asking again?"
        else:  # This means the revisor_node might not have been reached or its output wasn't captured
            logger.error(
                "VIEW: LangGraph stream finished, but final state from 'revisor_node' was not captured as expected.")
            ai_response_message = "An unexpected issue occurred with the AI assistant's processing pipeline."

    except Exception as e:
        logger.exception(f"VIEW: CRITICAL ERROR during LangGraph invocation or stream processing: {e}")
        ai_response_message = "Sorry, I encountered a critical error while processing your request. Please try again later."
        # If graph crashes, final_graph_output_state might be None.
        # We fallback to run_input_state to at least preserve the user's message in history.
        if final_graph_output_state is None:
            final_graph_output_state = run_input_state
            # Ensure chat_history from run_input_state is used for the session update
            # because final_graph_output_state was just overwritten.
            # The AI message will be the error message.

    # --- Post-Graph Processing ---

    # Ensure final_graph_output_state is a valid AgentState dictionary.
    # If the graph failed catastrophically, final_graph_output_state might still be None
    # or might not be the full state. We need to ensure chat_history is preserved from run_input_state at least.
    if final_graph_output_state is None:
        logger.error(
            "VIEW: final_graph_output_state is None after graph execution attempt. Using run_input_state as base.")
        final_graph_output_state = run_input_state  # Fallback to the state we sent to the graph
        # Ensure chat_history is a list if it somehow got corrupted
        if not isinstance(final_graph_output_state.get("chat_history"), list):
            final_graph_output_state["chat_history"] = []
    else:
        # If graph ran, final_graph_output_state should have its own chat_history.
        # This history would be the same as run_input_state's history because agents don't modify
        # the history list itself, they just read from it. The list is modified in the view.
        # So, we can directly use final_graph_output_state's chat_history.
        if not isinstance(final_graph_output_state.get("chat_history"), list):
            logger.warning(
                "VIEW: chat_history in final_graph_output_state is not a list. Using run_input_state's history.")
            final_graph_output_state["chat_history"] = list(run_input_state.get("chat_history", []))

    # **POINT 2: Add AI's final response to the chat_history that will be saved in the session and sent to frontend**
    # This `final_graph_output_state["chat_history"]` should contain the user's message already (from POINT 1).
    final_graph_output_state["chat_history"].append({"sender": "ai", "message": ai_response_message})
    logger.debug(
        f"VIEW: Appended AI response to final_graph_output_state.chat_history. New length: {len(final_graph_output_state['chat_history'])}")

    # --- Optional: Save successfully extracted OCR text to DB ---
    if final_graph_output_state.get("medical_document_id") and \
            final_graph_output_state.get("ocr_extracted_text") and \
            not final_graph_output_state.get("ocr_error_message"):  # Only save if OCR was successful
        try:
            doc_to_update = MedicalDocument.objects.get(id=final_graph_output_state["medical_document_id"])
            doc_to_update.raw_text = final_graph_output_state["ocr_extracted_text"]
            doc_to_update.save(update_fields=['raw_text'])
            logger.info(f"VIEW: Successfully saved OCR text to MedicalDocument ID {doc_to_update.id}")
        except MedicalDocument.DoesNotExist:  # Should not happen if ID came from a saved instance
            logger.error(
                f"VIEW: DB Save OCR - MedicalDocument ID {final_graph_output_state['medical_document_id']} not found during save attempt.")
        except Exception as e:
            logger.exception(f"VIEW: DB Save OCR - Error saving OCR text to MedicalDocument: {e}")

    # --- Save final state (including new user and AI messages in history) to session ---
    request.session['langgraph_medical_assistant_state_v2'] = final_graph_output_state
    request.session.modified = True
    logger.info("VIEW: Updated session state saved with new chat history.")

    logger.debug("--- VIEW: Exiting chat_api_endpoint (SUCCESS) ---")
    return JsonResponse({
        "status": "success",
        "message": ai_response_message,  # This is the AI's individual response text for the current turn
        "chat_history": final_graph_output_state["chat_history"]  # This is the full, updated chat log for the frontend
    })

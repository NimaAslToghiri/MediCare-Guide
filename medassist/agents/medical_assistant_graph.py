# DoctorAssistant/medassist/agents/medical_assistant_graph.py

import os
import base64
import logging
from typing import TypedDict, Optional, List, Dict

from django.conf import settings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # For converting history
from langgraph.graph import StateGraph, END

from medassist.models import MedicalDocument

# from pdf2image import convert_from_path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Configure LLMs (remains the same as your last version) ---
OPENAI_API_KEY = settings.OPENAI_API_KEY
llm_openai_reasoning_revising = None
llm_openai_vision_ocr = None
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set. LLMs will not be initialized.")
else:
    try:
        llm_openai_reasoning_revising = ChatOpenAI(
            model="gpt-3.5-turbo-0125", temperature=0.5,
            api_key=OPENAI_API_KEY, max_tokens=1500
        )
        logger.info("OpenAI LLM for Reasoning/Revising (gpt-3.5-turbo) initialized.")
        llm_openai_vision_ocr = ChatOpenAI(
            model="gpt-4o", temperature=0.2,
            api_key=OPENAI_API_KEY, max_tokens=2048
        )
        logger.info("OpenAI LLM for Vision/OCR (gpt-4o) initialized.")
    except Exception as e:
        logger.error(f"Error initializing OpenAI LLMs: {e}")


# --- LangGraph State Definition (remains the same) ---
class AgentState(TypedDict):
    user_id: int
    chat_history: List[
        Dict[str, str]]  # Format: [{"sender": "user", "message": "..."}, {"sender": "ai", "message": "..."}]
    current_user_input_text: Optional[str]
    medical_document_id: Optional[int]
    ocr_extracted_text: Optional[str]
    ocr_error_message: Optional[str]
    initial_reasoning_output: Optional[str]
    reasoning_error_message: Optional[str]
    final_llm_response: Optional[str]
    revisor_error_message: Optional[str]


# --- Helper function to prepare chat history for LLM ---
def format_chat_history_for_llm(history: List[Dict[str, str]], max_messages: int = 8) -> List:
    """
    Converts chat history from session format to LangChain message objects.
    Takes the last `max_messages`.
    """
    langchain_messages = []
    # Take the most recent messages
    recent_history = history[-max_messages:]
    for item in recent_history:
        sender = item.get("sender", "unknown").lower()
        message_text = item.get("message", "")
        if sender == "user":
            langchain_messages.append(HumanMessage(content=message_text))
        elif sender == "ai" or sender == "agent":  # Accommodate "agent" if used
            langchain_messages.append(AIMessage(content=message_text))
    return langchain_messages


# --- Agent Nodes ---

# OCR Agent (Agent 2 - remains the same as your last version)
def ocr_agent_openai(state: AgentState) -> AgentState:
    logger.debug("[AGENT OCR-OpenAI] === ENTERING ocr_agent_openai ===")
    new_state = state.copy()
    new_state["ocr_extracted_text"] = None
    new_state["ocr_error_message"] = None
    medical_doc_id = state.get("medical_document_id")

    if not llm_openai_vision_ocr:
        logger.error("[AGENT OCR-OpenAI] OpenAI Vision LLM not initialized.")
        new_state["ocr_error_message"] = "OCR service (OpenAI Vision) is not available."
        return new_state
    if not medical_doc_id:
        logger.debug("[AGENT OCR-OpenAI] No medical_document_id provided. Skipping OCR.")
        return new_state
    try:
        doc = MedicalDocument.objects.get(id=medical_doc_id)
        if not doc.file or not doc.file.name:
            new_state["ocr_error_message"] = "Document record found, but the file is missing."
            return new_state
        file_path = doc.file.path
        file_name = os.path.basename(doc.file.name)
        file_extension = os.path.splitext(file_name)[1].lower()
        image_messages_content = []
        if file_extension in [".jpg", ".jpeg", ".png", ".webp", ".gif"]:
            with open(file_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            image_messages_content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/{file_extension[1:]};base64,{base64_image}"}})
        elif file_extension == ".pdf":
            try:
                from pdf2image import convert_from_path  # Requires poppler
                images_from_pdf = convert_from_path(file_path, dpi=200, fmt='png', thread_count=2)  # Added thread_count
                if not images_from_pdf:
                    new_state["ocr_error_message"] = "Could not extract pages from PDF for OCR."
                    return new_state
                for i, page_image in enumerate(images_from_pdf):
                    if i >= 3:  # Limit pages for cost/performance
                        logger.info(f"Processing first 3 pages of PDF {file_name}.")
                        break
                    import io
                    buffered = io.BytesIO()
                    page_image.save(buffered, format="PNG")
                    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    image_messages_content.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})
                if not image_messages_content:
                    new_state["ocr_error_message"] = "No images processed from PDF."
                    return new_state
            except ImportError:
                new_state["ocr_error_message"] = "Cannot process PDF: Required library (pdf2image/Poppler) missing."
                return new_state
            except Exception as pdf_e:
                new_state["ocr_error_message"] = f"Error during PDF conversion: {str(pdf_e)[:100]}..."
                return new_state
        else:
            new_state["ocr_error_message"] = f"Unsupported file type ({file_extension})."
            return new_state
        if not image_messages_content:
            new_state["ocr_error_message"] = "Could not prepare document for OCR."
            return new_state
        ocr_prompt_text = "Extract all text content from the provided image(s). If it's a medical document or lab report, accurately transcribe all text, values, units, and medical terms. For multi-page documents (multiple images), combine extracted text sequentially."
        image_messages_content.insert(0, {"type": "text", "text": ocr_prompt_text})
        response = llm_openai_vision_ocr.invoke([{"role": "user", "content": image_messages_content}])
        extracted_text = response.content
        if extracted_text and extracted_text.strip():
            new_state["ocr_extracted_text"] = extracted_text.strip()
        else:
            new_state["ocr_error_message"] = "No text could be extracted by OCR."
    except Exception as e:
        logger.exception(f"[AGENT OCR-OpenAI] Unexpected error: {e}")
        new_state["ocr_error_message"] = f"Unexpected OCR error: {str(e)[:100]}..."
    logger.debug("[AGENT OCR-OpenAI] === EXITING ocr_agent_openai ===")
    return new_state


# Reasoning Agent (Agent 3 - Updated with Chat History)
def reasoning_agent(state: AgentState) -> AgentState:
    logger.debug("[AGENT REASONING] === ENTERING reasoning_agent ===")
    new_state = state.copy()
    new_state["initial_reasoning_output"] = None
    new_state["reasoning_error_message"] = None

    user_input_text = state.get("current_user_input_text")
    ocr_text = state.get("ocr_extracted_text")
    ocr_error = state.get("ocr_error_message")
    raw_chat_history = state.get("chat_history", [])

    # Prepare chat history for the LLM
    formatted_history = format_chat_history_for_llm(raw_chat_history, max_messages=6)  # Last 6 messages (3 turns)

    if not llm_openai_reasoning_revising:
        logger.error("[AGENT REASONING] Reasoning LLM not initialized.")
        new_state["reasoning_error_message"] = "Reasoning service is currently unavailable."
        return new_state

    reasoning_context_parts = []
    # Current query/document info is the primary focus for reasoning
    current_query_info = []
    if user_input_text and user_input_text.strip():
        current_query_info.append(f"User's latest query/message: \"{user_input_text.strip()}\"")
    if ocr_error:
        current_query_info.append(f"Note on latest document processing: {ocr_error}")
    elif ocr_text and ocr_text.strip():
        current_query_info.append(
            f"Extracted text from latest uploaded document:\n--- DOCUMENT START ---\n{ocr_text.strip()}\n--- DOCUMENT END ---")

    if not current_query_info and not formatted_history:  # No new input and no history
        logger.info("[AGENT REASONING] No actionable input or history for reasoning.")
        new_state[
            "initial_reasoning_output"] = "No specific information or conversation history available for detailed analysis."
        return new_state

    if current_query_info:
        reasoning_context_parts.append("Current User Input & Document Scan Analysis:")
        reasoning_context_parts.extend(current_query_info)
        reasoning_context_parts.append("\n---")  # Separator

    # System prompt for reasoning
    system_prompt_reasoning = """
You are an AI analytical engine. Your task is to deeply analyze the provided medical-related information. This includes the user's latest query/uploaded document AND the preceding conversation history if relevant.
Your goal is to:
1. Identify key medical information, symptoms, test results, or questions from the LATEST input and relevant parts of the conversation history.
2. Synthesize this information.
3. Identify potential patterns, anomalies, or areas that might warrant further attention or clarification (without making a diagnosis).
4. Structure your analysis clearly. This is an internal monologue or a detailed breakdown for another AI agent (a revisor) to use.
if user wants you speak in persian you have to speak in persian language with them until they asks you to speak in another language.
Do NOT provide medical advice, diagnoses, or treatment plans. Your output is for an internal AI review.
Focus on objective analysis. Acknowledge any document processing errors as limitations.
Output your detailed analysis.
"""

    # Construct messages for the prompt
    messages_for_prompt = [SystemMessage(content=system_prompt_reasoning)]
    messages_for_prompt.extend(formatted_history)  # Add past conversation
    # Add the current input as the latest human message IF there is new input.
    # If no new text/OCR, the reasoning will be based on history.
    if reasoning_context_parts:  # If there's new info (text or OCR)
        messages_for_prompt.append(HumanMessage(content="\n".join(reasoning_context_parts)))
    elif not formatted_history:  # No new info AND no history - should have been caught, but defensive
        new_state["initial_reasoning_output"] = "No information to analyze."
        return new_state

    prompt_template = ChatPromptTemplate.from_messages(messages_for_prompt)  # Create template from messages list
    # This is slightly different from using MessagesPlaceholder directly,
    # but achieves similar effect by constructing the message list.
    # For more dynamic history insertion, MessagesPlaceholder is cleaner.

    # Alternative with MessagesPlaceholder (cleaner for dynamic history):
    # prompt_template = ChatPromptTemplate.from_messages([
    #     SystemMessage(content=system_prompt_reasoning),
    #     MessagesPlaceholder(variable_name="formatted_chat_history"),
    #     HumanMessage(content="{current_input_summary}") # Pass the new info here
    # ])

    logger.debug(
        f"[AGENT REASONING] Context for reasoning (new + hist len {len(formatted_history)}): {messages_for_prompt[-1].content[:200]}...")

    try:
        # If using the direct message list approach:
        response_obj = llm_openai_reasoning_revising.invoke(messages_for_prompt)
        # If using MessagesPlaceholder approach:
        # response_obj = llm_openai_reasoning_revising.invoke({
        #     "formatted_chat_history": formatted_history,
        #     "current_input_summary": "\n".join(reasoning_context_parts) if reasoning_context_parts else "No new input for this turn, review history."
        # })

        reasoning_content = response_obj.content
        new_state["initial_reasoning_output"] = reasoning_content
        logger.info(f"[AGENT REASONING] Reasoning output generated (first 100 chars): {reasoning_content[:100]}...")
    except Exception as e:
        logger.exception(f"[AGENT REASONING] Error during reasoning LLM call: {e}")
        new_state["reasoning_error_message"] = f"Error during data analysis: {str(e)[:100]}..."
        new_state["initial_reasoning_output"] = "Failed to perform detailed analysis due to an internal error."

    logger.debug("[AGENT REASONING] === EXITING reasoning_agent ===")
    return new_state


# Revisor Agent (Agent 4 - Updated with Chat History and Persian Language Support)
def revisor_agent(state: AgentState) -> AgentState:
    logger.debug("[AGENT REVISOR] === ENTERING revisor_agent ===")
    new_state = state.copy()
    new_state["final_llm_response"] = None
    new_state["revisor_error_message"] = None

    user_input_text = state.get("current_user_input_text")  # Latest user text
    ocr_text = state.get("ocr_extracted_text")
    ocr_error = state.get("ocr_error_message")
    initial_analysis = state.get("initial_reasoning_output")
    reasoning_error = state.get("reasoning_error_message")
    raw_chat_history = state.get("chat_history", [])

    # Prepare chat history for the LLM (includes the latest user message if it was appended by view)
    # The last message in raw_chat_history here IS the current user's message.
    formatted_history_for_revisor = format_chat_history_for_llm(raw_chat_history,
                                                                max_messages=8)  # Last 8 messages (4 turns)

    if not llm_openai_reasoning_revising:  # Using the same LLM instance for revision
        logger.error("[AGENT REVISOR] Revising LLM not initialized.")
        new_state["final_llm_response"] = "Error: AI response generation service is unavailable."
        return new_state

    # Determine the language of the last user message for more explicit instruction
    # This is a simple check; a proper library would be more robust.
    # For now, we'll heavily rely on the LLM understanding the prompt.
    last_user_message_content = ""
    if formatted_history_for_revisor and isinstance(formatted_history_for_revisor[-1], HumanMessage):
        last_user_message_content = formatted_history_for_revisor[-1].content
    elif user_input_text:  # Fallback if history is empty but there's current input text
        last_user_message_content = user_input_text

    # Build the context string that the system prompt will refer to as "{input_for_revisor}"
    revisor_input_context_parts = []
    revisor_input_context_parts.append(
        "Current Interaction Context (Use this to understand the user's latest request and any document details):")
    if user_input_text and user_input_text.strip():
        revisor_input_context_parts.append(f"- User's most recent direct message: \"{user_input_text.strip()}\"")
    if ocr_error:
        revisor_input_context_parts.append(f"- Regarding an uploaded document: {ocr_error}")
    elif ocr_text and ocr_text.strip():
        revisor_input_context_parts.append(
            f"- Information extracted from their uploaded document (summary for your context): \"{ocr_text[:300]}...\"")

    if reasoning_error:
        revisor_input_context_parts.append(f"- An issue occurred during internal analysis: {reasoning_error}")
        revisor_input_context_parts.append(
            f"- Preliminary internal analysis was: {initial_analysis if initial_analysis else 'Not available due to error.'}")
    elif initial_analysis:
        revisor_input_context_parts.append(
            f"- Internal AI analysis notes (use these to inform your user-facing response, but rephrase into natural language):\n--- INTERNAL ANALYSIS ---\n{initial_analysis}\n--- END INTERNAL ANALYSIS ---")
    else:
        revisor_input_context_parts.append(
            "- No detailed internal analysis was performed for this turn (perhaps awaiting more specific input or due to prior errors).")

    final_input_for_llm_prompt = "\n".join(revisor_input_context_parts)
    logger.debug(
        f"[AGENT REVISOR] Input context for revisor prompt (first 300 chars): {final_input_for_llm_prompt[:300]}...")

    # --- UPDATED SYSTEM PROMPT with more explicit language instructions ---
    system_prompt_revisor_and_responder = f"""
You are "MediCare Guide", a specialized AI medical assistant. Your primary purpose is to provide guidance, information, and support related to health and medical topics.
You will be given the recent conversation history (if any) and a summary of the current interaction (user's latest message, document info, and internal analysis notes).
Based on ALL this information, generate a helpful, empathetic, and clear response FOR THE USER.

**LANGUAGE DETERMINATION AND EXECUTION (VERY IMPORTANT):**
1.  **Assess User's Language:** Carefully examine the user's LATEST message (provided in the 'Current Interaction Context' below, and also as the last message in the chat history). Also, check if Persian was used or requested in recent chat history.
2.  **Switch to Persian:** If the user's LATEST message is predominantly in Persian, OR if the user has explicitly requested to converse in Persian (e.g., by saying "به فارسی صحبت کن" or similar, either now or in recent history), then your ENTIRE response MUST be in fluent, natural, and grammatically correct Persian.
3.  **English Default:** Otherwise, your ENTIRE response MUST be in English.
4.  **Consistency is Key:** Once you determine the language for the current response (Persian or English), ALL parts of your response – greetings, the main answer, any questions you ask the user, and standard phrases (like disclaimers or builder information if triggered) – MUST be in that chosen language. Do not mix languages within a single response.
5.  **Maintain Language Context:** If the conversation has clearly established Persian as the language of interaction, continue in Persian for subsequent turns unless the user initiates a switch back to English (e.g., by typing in English).

**Pre-defined Phrases (use the version matching the determined response language):**
* **Builder Information:** If specifically asked "who built you" or "who is your builder" (or similar queries about your origin):
    * English: "I am an AI assistant developed by the StackToServ team, dedicated to building intelligent AI systems."
    * Persian: "من یک دستیار هوش مصنوعی هستم که توسط تیم StackToServ توسعه داده شده‌ام؛ تیمی که به ساخت سیستم‌های هوشمند هوش مصنوعی اختصاص دارد."
    * (Do not volunteer this information otherwise.)
* **Medical Disclaimer (MANDATORY in every relevant response):**
    * English: "Please remember, this is AI-generated information, not a substitute for professional medical advice. Always consult with a doctor or qualified healthcare provider for any health concerns or before making any decisions related to your health."
    * Persian: "لطفا به یاد داشته باشید، این اطلاعات توسط هوش مصنوعی تولید شده و جایگزین توصیه پزشکی حرفه‌ای نیست. همیشه برای هرگونه نگرانی سلامتی یا قبل از تصمیم‌گیری در مورد سلامتی خود با پزشک یا ارائه‌دهنده خدمات بهداشتی واجد شرایط مشورت کنید."

**Response Guidelines (apply these in the chosen language):**
1.  **Acknowledge & Synthesize:** Briefly acknowledge the user's latest input. Use insights from the internal analysis (if provided) to inform your response, rephrasing it into user-friendly language. Do NOT directly quote internal analysis notes.
2.  **Handle Errors:** If document/analysis errors are noted, politely inform the user.
3.  **Maintain Persona:** Always respond as "MediCare Guide".
4.  **Focus on Health:** Stick to health topics. If the LATEST user query seems off-topic, gently redirect.
5.  **Professional & Empathetic Tone.**
6.  **CRITICAL - No Medical Advice (Reiterate):** You are an AI. You cannot provide diagnoses, treatment plans, or prescriptions. Reinforce this and ALWAYS include the appropriate language version of the Medical Disclaimer above.
7.  **Conversational Flow:** Refer to relevant points from the `chat_history` if it helps maintain context or answer follow-up questions naturally.
8.  **Readability:** Use paragraphs and bullets if helpful.

If the current input context is very generic, lacks medical focus, or if previous attempts to guide the user to medical topics failed, provide a general greeting in the determined language and offer assistance with medical topics. For example, (English: "Hello! I'm MediCare Guide. How can I help you with your medical questions or documents today?") (Persian: "سلام! من «مدی‌کر گاید» هستم. امروز چطور می‌توانم در مورد سوالات پزشکی یا مدارک شما کمک کنم؟").

User's LATEST message content for your immediate attention: "{last_user_message_content}"
"""
    # Note: last_user_message_content is added to the system prompt for emphasis on the latest user input language.
    # The full context for the 'human' part of the prompt is still final_input_for_llm_prompt

    messages_for_llm = [SystemMessage(content=system_prompt_revisor_and_responder)]
    messages_for_llm.extend(formatted_history_for_revisor)  # Add recent chat history
    messages_for_llm.append(HumanMessage(content=final_input_for_llm_prompt))  # Current turn's detailed context

    try:
        response_obj = llm_openai_reasoning_revising.invoke(messages_for_llm)
        final_content = response_obj.content
        new_state["final_llm_response"] = final_content
        logger.info(f"[AGENT REVISOR] Final response generated (first 100 chars): {final_content[:100]}...")
    except Exception as e:
        logger.exception(f"[AGENT REVISOR] Error during final response generation: {e}")
        new_state["revisor_error_message"] = f"Error generating final AI reply: {str(e)[:100]}..."
        new_state[
            "final_llm_response"] = "I apologize, but I encountered an issue while processing your request. Please try again."

    logger.debug("[AGENT REVISOR] === EXITING revisor_agent ===")
    return new_state


# --- Define the LangGraph Workflow (remains the same) ---
workflow = StateGraph(AgentState)
workflow.add_node("ocr_node", ocr_agent_openai)
workflow.add_node("reasoning_node", reasoning_agent)
workflow.add_node("revisor_node", revisor_agent)
workflow.set_entry_point("ocr_node")
workflow.add_edge("ocr_node", "reasoning_node")
workflow.add_edge("reasoning_node", "revisor_node")
workflow.add_edge("revisor_node", END)
try:
    compiled_medical_assistant_graph = workflow.compile()
    logger.info("LangGraph with OCR, Reasoning, and Revisor agents compiled.")
except Exception as e:
    logger.exception("Failed to compile LangGraph:")
    compiled_medical_assistant_graph = None
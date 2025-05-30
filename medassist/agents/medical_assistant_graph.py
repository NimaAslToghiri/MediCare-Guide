# DoctorAssistant/medassist/agents/medical_assistant_graph.py

import os
import base64
import logging
import pickle  # For generic pickle loading if FAISS.load_local isn't a direct match
from typing import TypedDict, Optional, List, Dict, Any

from django.conf import settings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from medassist.models import MedicalDocument
from DoctorAssistant.settings import OPENAI_API_KEY


# from pdf2image import convert_from_path # If PDF processing enabled

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Attempt to import FAISS
try:
    from langchain_community.vectorstores import FAISS
    import faiss  # Ensure faiss is installed

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS or langchain_community not fully available. RAG will be disabled.")



# --- RAG Configuration ---
# Define path to your RAG index files.
# IMPORTANT: Adjust this path to where your 'index.pkl' and 'index.faiss' (if using FAISS) are stored.
# It's usually a directory. LangChain's FAISS.load_local expects a folder_path and an index_name (default is "index").
# If your 'index.pkl' is a standalone pickle of the entire vector store object, loading is different.
RAG_INDEX_FOLDER_PATH = os.path.join(settings.BASE_DIR, "rag_data")  # Assumes rag_data is in your project root
RAG_INDEX_NAME = "index"  # Default for FAISS, 'index.pkl' and 'index.faiss' will be looked for

vector_store = None
embeddings_model_for_rag = None

if FAISS_AVAILABLE and OPENAI_API_KEY:
    try:
        embeddings_model_for_rag = OpenAIEmbeddings(
            model="text-embedding-3-small",  # Use the same model used to create the embeddings
            api_key=OPENAI_API_KEY
        )
        logger.info("OpenAI Embeddings model for RAG initialized ('text-embedding-3-small').")

        # Check if the specific index files exist before attempting to load
        faiss_file = os.path.join(RAG_INDEX_FOLDER_PATH, f"{RAG_INDEX_NAME}.faiss")
        pkl_file = os.path.join(RAG_INDEX_FOLDER_PATH, f"{RAG_INDEX_NAME}.pkl")

        if os.path.exists(faiss_file) and os.path.exists(pkl_file):
            vector_store = FAISS.load_local(
                folder_path=RAG_INDEX_FOLDER_PATH,
                embeddings=embeddings_model_for_rag,
                index_name=RAG_INDEX_NAME,
                allow_dangerous_deserialization=True  # Required by LangChain for loading pickled custom Python objects
            )
            logger.info(
                f"FAISS vector store loaded successfully from {RAG_INDEX_FOLDER_PATH} with index name '{RAG_INDEX_NAME}'.")
        else:
            logger.error(
                f"RAG index files not found at {RAG_INDEX_FOLDER_PATH} (expected {RAG_INDEX_NAME}.faiss and {RAG_INDEX_NAME}.pkl). RAG will be disabled.")
            vector_store = None  # Ensure it's None if loading fails
    except Exception as e:
        logger.exception(
            f"Error loading FAISS vector store or initializing embeddings model for RAG: {e}. RAG will be disabled.")
        vector_store = None
else:
    if not FAISS_AVAILABLE:
        logger.warning("FAISS library not available. RAG features will be disabled.")
    if not OPENAI_API_KEY:
        logger.warning("OpenAI API Key not available. RAG embeddings model cannot be initialized.")
    vector_store = None

# --- Configure LLMs (as before) ---
llm_openai_reasoning_revising = None
llm_openai_vision_ocr = None
# ... (rest of your LLM initialization code from the previous version, ensure it's present) ...
if not OPENAI_API_KEY:  # This was for logging, actual init is below
    logger.error("OPENAI_API_KEY is not set in Django settings. OpenAI LLMs will not be initialized.")
else:
    try:
        llm_openai_reasoning_revising = ChatOpenAI(
            model="gpt-3.5-turbo-0125", temperature=0.5,
            api_key=OPENAI_API_KEY, max_tokens=1500
        )
        logger.info("OpenAI LLM for Reasoning/Revising (gpt-3.5-turbo) initialized successfully.")
        llm_openai_vision_ocr = ChatOpenAI(
            model="gpt-4o", temperature=0.2,
            api_key=OPENAI_API_KEY, max_tokens=2048
        )
        logger.info("OpenAI LLM for Vision/OCR (gpt-4o) initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing OpenAI LLMs: {e}")


# --- LangGraph State Definition ---
class AgentState(TypedDict):  # Expanded for RAG
    user_id: int
    chat_history: List[Dict[str, str]]
    current_user_input_text: Optional[str]
    medical_document_id: Optional[int]

    ocr_extracted_text: Optional[str]
    ocr_error_message: Optional[str]

    # For RAG process
    query_for_rag: Optional[str]  # The (potentially translated) query used for retrieval
    translated_query_for_rag: Optional[str]  # If translation happened
    retrieved_rag_documents: Optional[List[Dict[str, Any]]]  # List of {page_content: str, metadata: dict}

    initial_reasoning_output: Optional[str]
    reasoning_error_message: Optional[str]

    final_llm_response: Optional[str]
    revisor_error_message: Optional[str]


# --- Helper function for chat history (as before) ---
def format_chat_history_for_llm(history: List[Dict[str, str]], max_messages: int = 8) -> List:
    # ... (implementation from previous version) ...
    langchain_messages = []
    recent_history = history[-max_messages:]
    for item in recent_history:
        sender = item.get("sender", "unknown").lower()
        message_text = item.get("message", "")
        if sender == "user":
            langchain_messages.append(HumanMessage(content=message_text))
        elif sender == "ai" or sender == "agent":
            langchain_messages.append(AIMessage(content=message_text))
    return langchain_messages


# --- Agent Nodes ---

# OCR Agent (Agent 2 - ocr_agent_openai - as before)
# ... (full code for ocr_agent_openai from the previous version) ...
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
                images_from_pdf = convert_from_path(file_path, dpi=200, fmt='png', thread_count=2)
                if not images_from_pdf:
                    new_state["ocr_error_message"] = "Could not extract pages from PDF for OCR."
                    return new_state
                for i, page_image in enumerate(images_from_pdf):
                    if i >= 3:
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


# Reasoning Agent (Agent 3 - Updated with RAG)
def reasoning_agent(state: AgentState) -> AgentState:
    logger.debug("[AGENT REASONING-RAG] === ENTERING reasoning_agent ===")
    new_state = state.copy()
    new_state["initial_reasoning_output"] = None
    new_state["reasoning_error_message"] = None
    new_state["query_for_rag"] = None
    new_state["translated_query_for_rag"] = None
    new_state["retrieved_rag_documents"] = []

    user_input_text = state.get("current_user_input_text")
    ocr_text = state.get("ocr_extracted_text")
    # ocr_error = state.get("ocr_error_message") # Already handled by ocr_agent, ocr_text will be None or error string
    raw_chat_history = state.get("chat_history", [])
    formatted_history = format_chat_history_for_llm(raw_chat_history, max_messages=6)

    if not llm_openai_reasoning_revising:
        logger.error("[AGENT REASONING-RAG] Reasoning LLM not initialized.")
        new_state["reasoning_error_message"] = "Reasoning service is unavailable."
        return new_state

    # 1. Consolidate query for RAG from current user input and OCR
    query_parts_for_rag = []
    if user_input_text and user_input_text.strip():
        query_parts_for_rag.append(user_input_text.strip())
    if ocr_text and ocr_text.strip() and "Error:" not in ocr_text and "No text could be extracted" not in ocr_text:  # Only use valid OCR text
        query_parts_for_rag.append(ocr_text.strip())

    consolidated_query_for_rag = "\n".join(query_parts_for_rag).strip()
    new_state["query_for_rag"] = consolidated_query_for_rag
    query_for_embedding_search = consolidated_query_for_rag  # This will be the English query

    # 2. Translate consolidated query to English if it seems to be Persian (for RAG search)
    # This is a simplified check. A robust solution might use a language detection library.
    # We'll ask the LLM to translate if the query is not empty and seems non-English or if user has indicated Persian.
    # For simplicity, let's assume if the user's *last message* has Persian characters, we attempt translation.
    # A more direct check: if `user_input_text` contains Persian characters.
    # This is a heuristic.

    # Basic check for Persian characters (very naive, expand with more chars if needed)
    is_persian_heuristic = any(u'\u0600' <= char <= u'\u06FF' for char in consolidated_query_for_rag)

    if consolidated_query_for_rag and is_persian_heuristic:
        logger.info("[AGENT REASONING-RAG] Original query appears to be in Persian. Translating to English for RAG.")
        try:
            translate_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(
                    content="You are a highly proficient translator. Translate the following text accurately to English. Output ONLY the translated English text and nothing else."),
                HumanMessage(content=consolidated_query_for_rag)
            ])
            translation_response = llm_openai_reasoning_revising.invoke(translate_prompt.format_messages())
            query_for_embedding_search = translation_response.content.strip()
            new_state["translated_query_for_rag"] = query_for_embedding_search
            logger.info(f"[AGENT REASONING-RAG] Translated query for RAG: {query_for_embedding_search[:100]}...")
            if not query_for_embedding_search:  # Handle empty translation
                logger.warning(
                    "[AGENT REASONING-RAG] Translation to English resulted in empty string. Using original query for RAG if available.")
                query_for_embedding_search = consolidated_query_for_rag  # Fallback
        except Exception as trans_e:
            logger.exception(
                f"[AGENT REASONING-RAG] Error translating query to English: {trans_e}. Using original query for RAG.")
            # Fallback to original query if translation fails
            query_for_embedding_search = consolidated_query_for_rag

    # 3. Perform RAG search if a query exists and vector store is available
    retrieved_docs_content = []
    if query_for_embedding_search and vector_store and embeddings_model_for_rag:
        logger.info(f"[AGENT REASONING-RAG] Performing RAG search with query: {query_for_embedding_search[:100]}...")
        try:
            # FAISS.similarity_search returns a list of Document objects
            # A Document object has `page_content` and `metadata`
            retrieved_documents = vector_store.similarity_search(query_for_embedding_search, k=3)  # Get top 3 docs
            if retrieved_documents:
                new_state["retrieved_rag_documents"] = [{"page_content": doc.page_content, "metadata": doc.metadata} for
                                                        doc in retrieved_documents]
                for i, doc in enumerate(retrieved_documents):
                    retrieved_docs_content.append(
                        f"Retrieved Document {i + 1} (source: {doc.metadata.get('source', 'N/A')}):\n{doc.page_content}")
                logger.info(f"[AGENT REASONING-RAG] Retrieved {len(retrieved_documents)} documents for RAG.")
            else:
                logger.info("[AGENT REASONING-RAG] No relevant documents found by RAG.")
        except Exception as rag_e:
            logger.exception(f"[AGENT REASONING-RAG] Error during RAG similarity search: {rag_e}")
            # Continue without RAG documents if search fails

    # 4. Prepare context for the main reasoning LLM
    reasoning_llm_input_parts = []
    # Original user query and OCR (as before)
    current_query_info_for_llm = []
    if user_input_text and user_input_text.strip():
        current_query_info_for_llm.append(f"User's original latest query/message: \"{user_input_text.strip()}\"")
    ocr_error = state.get("ocr_error_message")  # Get it from the state set by OCR agent
    if ocr_error:
        current_query_info_for_llm.append(f"Note on document processing for the user's query: {ocr_error}")
    elif ocr_text and ocr_text.strip():
        current_query_info_for_llm.append(
            f"Extracted text from user's uploaded document:\n--- DOCUMENT START ---\n{ocr_text.strip()}\n--- DOCUMENT END ---")

    if not current_query_info_for_llm and not formatted_history and not retrieved_docs_content:
        logger.info("[AGENT REASONING-RAG] No actionable input, history, or RAG docs for reasoning.")
        new_state[
            "initial_reasoning_output"] = "No specific information was provided or retrieved for detailed analysis."
        return new_state

    if current_query_info_for_llm:
        reasoning_llm_input_parts.append(
            "Current User Input & Document Scan Details (Original Language if applicable):")
        reasoning_llm_input_parts.extend(current_query_info_for_llm)
        reasoning_llm_input_parts.append("\n---")

    # Add retrieved RAG documents if any
    if retrieved_docs_content:
        reasoning_llm_input_parts.append("Relevant Information from Medical Knowledge Base (in English):")
        reasoning_llm_input_parts.extend(retrieved_docs_content)
        reasoning_llm_input_parts.append("\n---")
    else:
        reasoning_llm_input_parts.append(
            "No specific information was retrieved from the medical knowledge base for this query.")
        reasoning_llm_input_parts.append("\n---")

    # System prompt for reasoning (augmented with RAG context)
    system_prompt_reasoning_rag = """
You are an AI analytical engine. Your task is to deeply analyze the provided medical-related information. This includes:
1. The user's latest query/uploaded document (in its original language).
2. Relevant preceding conversation history (if provided).
3. Relevant excerpts from a medical knowledge base (provided in English, if any were found).

Your goal is to:
- Synthesize all this information.
- Identify key medical facts, symptoms, test results, or questions.
- If knowledge base excerpts are provided, use them to inform your analysis. Note that these are general medical texts.
- Identify potential patterns, anomalies, or areas that might warrant further attention or clarification (without making a diagnosis).
- Structure your analysis clearly. It should be an internal monologue or a detailed breakdown that another AI agent (a revisor) will use to formulate a user-friendly response.
- Your analysis output should primarily be in English to facilitate the next AI review step, especially when incorporating English knowledge base excerpts.
Do NOT provide medical advice, diagnoses, or treatment plans directly in this output. Your output is for an internal AI review.
If there were errors in document processing or RAG retrieval, acknowledge them as limitations.
Output your detailed analysis.
"""
    messages_for_prompt = [SystemMessage(content=system_prompt_reasoning_rag)]
    messages_for_prompt.extend(formatted_history)  # Add past conversation

    final_reasoning_context_for_llm = "\n".join(reasoning_llm_input_parts)
    if final_reasoning_context_for_llm:
        messages_for_prompt.append(HumanMessage(content=final_reasoning_context_for_llm))
    elif not formatted_history:  # No new info AND no history - should have been caught, but defensive
        new_state["initial_reasoning_output"] = "No information to analyze."
        return new_state

    logger.debug(
        f"[AGENT REASONING-RAG] Final context for reasoning LLM (hist len {len(formatted_history)}): {messages_for_prompt[-1].content[:200]}...")

    try:
        response_obj = llm_openai_reasoning_revising.invoke(messages_for_prompt)
        reasoning_content = response_obj.content
        new_state["initial_reasoning_output"] = reasoning_content
        logger.info(f"[AGENT REASONING-RAG] Reasoning output generated (first 100 chars): {reasoning_content[:100]}...")
    except Exception as e:
        logger.exception(f"[AGENT REASONING-RAG] Error during reasoning LLM call: {e}")
        new_state["reasoning_error_message"] = f"Error during data analysis: {str(e)[:100]}..."
        new_state["initial_reasoning_output"] = "Failed to perform detailed analysis due to an internal error."

    logger.debug("[AGENT REASONING-RAG] === EXITING reasoning_agent ===")
    return new_state


# Revisor Agent (Agent 4 - revisor_agent - as before, but now uses RAG-informed reasoning)
# ... (full code for revisor_agent from the previous version, its logic to use initial_reasoning_output and handle language remains the same) ...
def revisor_agent(state: AgentState) -> AgentState:
    logger.debug("[AGENT REVISOR] === ENTERING revisor_agent ===")
    new_state = state.copy()
    new_state["final_llm_response"] = None
    new_state["revisor_error_message"] = None

    user_input_text = state.get("current_user_input_text")
    ocr_text = state.get("ocr_extracted_text")
    ocr_error = state.get("ocr_error_message")
    initial_analysis = state.get("initial_reasoning_output")
    reasoning_error = state.get("reasoning_error_message")
    retrieved_rag_docs = state.get("retrieved_rag_documents")  # For context if needed by revisor
    raw_chat_history = state.get("chat_history", [])
    formatted_history_for_revisor = format_chat_history_for_llm(raw_chat_history, max_messages=8)

    if not llm_openai_reasoning_revising:
        logger.error("[AGENT REVISOR] Revising LLM not initialized.")
        new_state["final_llm_response"] = "Error: AI response generation service is unavailable."
        return new_state

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

    if retrieved_rag_docs:
        revisor_input_context_parts.append(
            f"- The system retrieved {len(retrieved_rag_docs)} relevant snippets from its knowledge base to inform the analysis.")
        # Could optionally include snippets here if short, or just note they were used by reasoning.

    if reasoning_error:
        revisor_input_context_parts.append(f"- An issue occurred during internal analysis: {reasoning_error}")
        revisor_input_context_parts.append(
            f"- Preliminary internal analysis was: {initial_analysis if initial_analysis else 'Not available due to error.'}")
    elif initial_analysis:
        revisor_input_context_parts.append(
            f"- Internal AI analysis notes (use these to craft the user response, but rephrase into natural language):\n--- INTERNAL ANALYSIS ---\n{initial_analysis}\n--- END INTERNAL ANALYSIS ---")
    else:
        revisor_input_context_parts.append("- No detailed internal analysis was performed for this turn.")

    final_input_for_llm_prompt = "\n".join(revisor_input_context_parts)
    last_user_message_content = ""
    if formatted_history_for_revisor and isinstance(formatted_history_for_revisor[-1], HumanMessage):
        last_user_message_content = formatted_history_for_revisor[-1].content
    elif user_input_text:
        last_user_message_content = user_input_text

    system_prompt_revisor_and_responder = f"""
You are "MediCare Guide", a specialized AI medical assistant. Your primary purpose is to provide guidance, information, and support related to health and medical topics.
You will be given the recent conversation history (if any) and a summary of the current interaction (user's latest message, document info, and internal analysis notes which may be based on a knowledge base).
Based on ALL this information, generate a helpful, empathetic, and clear response FOR THE USER.

**LANGUAGE DETERMINATION AND EXECUTION (VERY IMPORTANT):**
1.  **Assess User's Language:** Carefully examine the user's LATEST message (content provided below, and also as the last message in the chat history). Also, check if Persian was used or requested in recent chat history.
2.  **Switch to Persian:** If the user's LATEST message is predominantly in Persian, OR if the user has explicitly requested to converse in Persian, then your ENTIRE response MUST be in fluent, natural, and grammatically correct Persian.
3.  **English Default:** Otherwise, your ENTIRE response MUST be in English.
4.  **Consistency is Key:** Once you determine the language, ALL parts of your response – greetings, the main answer, questions, and standard phrases (disclaimers, builder info if triggered) – MUST be in that chosen language.
5.  **Maintain Language Context:** If Persian is established, continue in Persian unless the user initiates a switch back to English.

**Pre-defined Phrases (use the version matching the determined response language):**
* **Builder Information:** If asked "who built you":
    * English: "I am an AI assistant developed by the StackToServ team, dedicated to building intelligent AI systems."
    * Persian: "من یک دستیار هوش مصنوعی هستم که توسط تیم StackToServ توسعه داده شده‌ام؛ تیمی که به ساخت سیستم‌های هوشمند هوش مصنوعی اختصاص دارد."
* **Medical Disclaimer (MANDATORY in every relevant response):**
    * English: "Please remember, this is AI-generated information, not a substitute for professional medical advice. Always consult with a doctor or qualified healthcare provider for any health concerns or before making any decisions related to your health."
    * Persian: "لطفا به یاد داشته باشید، این اطلاعات توسط هوش مصنوعی تولید شده و جایگزین توصیه پزشکی حرفه‌ای نیست. همیشه برای هرگونه نگرانی سلامتی یا قبل از تصمیم‌گیری در مورد سلامتی خود با پزشک یا ارائه‌دهنده خدمات بهداشتی واجد شرایط مشورت کنید."

**Response Guidelines (apply these in the chosen language):**
1.  **Acknowledge & Synthesize:** Briefly acknowledge the user's latest input. Use insights from the internal analysis (which may be RAG-informed) to inform your response, rephrasing it into user-friendly language.
2.  **Handle Errors:** If document/analysis/RAG errors are noted, politely inform the user.
3.  **Maintain Persona & Focus:** As "MediCare Guide", stick to health topics. Gently redirect if the LATEST query is off-topic.
4.  **CRITICAL - No Medical Advice:** You are an AI. You cannot provide diagnoses, treatment plans, or prescriptions. Reinforce this and ALWAYS include the appropriate language version of the Medical Disclaimer.
5.  **Conversational Flow:** Refer to relevant points from `chat_history` if it helps.

User's LATEST message content for your immediate attention on language: "{last_user_message_content}"
---
Now, considering all the above, and the following detailed context for the current turn, generate your response to the user:
"""
    messages_for_llm = [SystemMessage(content=system_prompt_revisor_and_responder)]
    messages_for_llm.extend(formatted_history_for_revisor)
    messages_for_llm.append(HumanMessage(content=final_input_for_llm_prompt))

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
workflow.add_node("reasoning_node", reasoning_agent)  # Now RAG-enabled
workflow.add_node("revisor_node", revisor_agent)
workflow.set_entry_point("ocr_node")
workflow.add_edge("ocr_node", "reasoning_node")
workflow.add_edge("reasoning_node", "revisor_node")
workflow.add_edge("revisor_node", END)
try:
    compiled_medical_assistant_graph = workflow.compile()
    logger.info("LangGraph with RAG-enabled Reasoning agent compiled successfully.")
except Exception as e:
    logger.exception("Failed to compile LangGraph with RAG updates:")
    compiled_medical_assistant_graph = None
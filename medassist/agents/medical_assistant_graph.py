# DoctorAssistant/medassist/agents/medical_assistant_graph.py

import os
import base64
import logging
from typing import TypedDict, Optional, List, Dict, Any

from django.conf import settings
from langchain_openai import ChatOpenAI  # OpenAIEmbeddings is no longer needed for RAG if using HuggingFace for it
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from medassist.models import MedicalDocument

# from pdf2image import convert_from_path # If PDF processing enabled

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(module)s:%(lineno)d - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Attempt to import FAISS and HuggingFaceEmbeddings
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    import faiss

    FAISS_HF_AVAILABLE = True
    logger.info("FAISS, HuggingFaceEmbeddings, and langchain_community successfully imported.")
except ImportError:
    FAISS_HF_AVAILABLE = False
    logger.warning("FAISS, HuggingFaceEmbeddings, or langchain_community not fully available. RAG will be disabled.")

# --- RAG Configuration ---
RAG_INDEX_FOLDER_PATH = os.path.join(settings.BASE_DIR, "rag_data")
RAG_INDEX_NAME = "index"  # Default base name for index.faiss and index.pkl

vector_store: Optional[FAISS] = None  # Type hinting for clarity
embeddings_model_for_rag: Optional[HuggingFaceEmbeddings] = None  # Changed type hint

logger.info("--- Starting RAG Configuration ---")
# OpenAI API Key is still needed for the Chat LLMs, but not for HuggingFace Embeddings by default
if FAISS_HF_AVAILABLE:  # Only need FAISS and HuggingFace for this RAG setup
    try:
        # Define hf_model_name and hf_model_kwargs HERE, BEFORE they are used
        hf_model_name = "sentence-transformers/all-mpnet-base-v2"
        hf_model_kwargs = {"device": "cpu"}  # Explicitly CPU for broader compatibility

        logger.info(
            f"Attempting to initialize HuggingFaceEmbeddings with model: '{hf_model_name}' on device: '{hf_model_kwargs['device']}'")

        embeddings_model_for_rag = HuggingFaceEmbeddings(
            model_name=hf_model_name,  # Use the defined variable
            model_kwargs=hf_model_kwargs,  # Use the defined variable
            # encode_kwargs={'normalize_embeddings': False} # Usually not needed for all-mpnet-base-v2 with FAISS
        )
        # Now log success using the defined variable
        logger.info(f"Initialized HuggingFace embedding model for RAG: '{hf_model_name}'")  # Uses the defined variable

        faiss_file_path = os.path.join(RAG_INDEX_FOLDER_PATH, f"{RAG_INDEX_NAME}.faiss")
        pkl_file_path = os.path.join(RAG_INDEX_FOLDER_PATH, f"{RAG_INDEX_NAME}.pkl")

        if os.path.exists(faiss_file_path) and os.path.exists(pkl_file_path):
            logger.info(
                f"Attempting to load FAISS index from folder: '{RAG_INDEX_FOLDER_PATH}' with index name: '{RAG_INDEX_NAME}'")
            vector_store = FAISS.load_local(
                folder_path=RAG_INDEX_FOLDER_PATH,
                embeddings=embeddings_model_for_rag,
                index_name=RAG_INDEX_NAME,
                allow_dangerous_deserialization=True
            )
            logger.info("FAISS vector store loaded successfully from disk.")
            if hasattr(vector_store, 'index') and vector_store.index:
                logger.info(f"IMPORTANT - Loaded FAISS index dimension (d_index / self.d): {vector_store.index.d}")
            else:
                logger.error(
                    "FAISS vector store loaded, but its '.index' attribute is missing or None. Cannot confirm stored vector dimension. RAG might fail.")
                vector_store = None
        else:
            logger.error(
                f"RAG index files ('{RAG_INDEX_NAME}.faiss' and/or '{RAG_INDEX_NAME}.pkl') not found in '{RAG_INDEX_FOLDER_PATH}'. RAG will be disabled.")
            vector_store = None
    except Exception as e:
        logger.exception(f"CRITICAL ERROR during RAG setup (HuggingFace Embeddings init or FAISS load): {e}")
        vector_store = None
else:
    if not FAISS_HF_AVAILABLE:
        logger.warning("FAISS or HuggingFaceEmbeddings library not available at startup. RAG features disabled.")
    vector_store = None
logger.info("--- Finished RAG Configuration ---")

# --- Configure LLMs (OpenAI for Chat, Vision) ---
llm_openai_reasoning_revising: Optional[ChatOpenAI] = None
llm_openai_vision_ocr: Optional[ChatOpenAI] = None

if settings.OPENAI_API_KEY:
    try:
        llm_openai_reasoning_revising = ChatOpenAI(
            model="gpt-4o",
            temperature=0.5,
            api_key=settings.OPENAI_API_KEY,
            max_tokens=1500
        )
        logger.info(f"OpenAI LLM for Reasoning/Revising ({llm_openai_reasoning_revising.model_name}) initialized.")

        llm_openai_vision_ocr = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            api_key=settings.OPENAI_API_KEY,
            max_tokens=2048
        )
        logger.info(f"OpenAI LLM for Vision/OCR ({llm_openai_vision_ocr.model_name}) initialized.")
    except Exception as e:
        logger.error(f"Error initializing OpenAI LLMs: {e}")
else:
    logger.error("OPENAI_API_KEY is not set. Cannot initialize main Chat LLMs.")


# --- LangGraph State Definition (remains the same as your last version) ---
class AgentState(TypedDict):
    user_id: int
    chat_history: List[Dict[str, str]]
    current_user_input_text: Optional[str]
    medical_document_id: Optional[int]
    ocr_extracted_text: Optional[str]
    ocr_error_message: Optional[str]
    query_for_rag: Optional[str]
    translated_query_for_rag: Optional[str]
    retrieved_rag_documents: Optional[List[Dict[str, Any]]]
    initial_reasoning_output: Optional[str]
    reasoning_error_message: Optional[str]
    final_llm_response: Optional[str]
    revisor_error_message: Optional[str]


# --- Helper function for chat history (remains the same) ---
def format_chat_history_for_llm(history: List[Dict[str, str]], max_messages: int = 8) -> List:
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

# OCR Agent (ocr_agent_openai - remains the same as your last version)
# ... (Insert the full code for ocr_agent_openai here, it doesn't change for this RAG embedding model fix)
def ocr_agent_openai(state: AgentState) -> AgentState:
    logger.debug("[AGENT OCR-OpenAI] === ENTERING ocr_agent_openai ===")
    new_state = state.copy()
    new_state["ocr_extracted_text"] = None
    new_state["ocr_error_message"] = None
    medical_doc_id = state.get("medical_document_id")

    if not llm_openai_vision_ocr:
        logger.error("[AGENT OCR-OpenAI] OpenAI Vision LLM not initialized. Skipping OCR.")
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

        if file_extension in [".jpg", ".jpeg", ".png", ".webp", ".gif"]:  # Common image types GPT-4V supports
            logger.info(f"[AGENT OCR-OpenAI] Processing image file: {file_name}")
            with open(file_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            image_messages_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{file_extension[1:]};base64,{base64_image}"}
            })
        elif file_extension == ".pdf":
            logger.info(f"[AGENT OCR-OpenAI] Processing PDF file: {file_name}.")
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
                    page_image.save(buffered, format="PNG")  # Convert page to PNG
                    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    image_messages_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    })
                if not image_messages_content:
                    new_state["ocr_error_message"] = "No images processed from PDF."
                    return new_state
            except ImportError:
                logger.error(
                    "[AGENT OCR-OpenAI] `pdf2image` or Poppler missing. PDF OCR via image conversion disabled.")
                new_state[
                    "ocr_error_message"] = "Cannot process PDF for OCR: Required library (pdf2image/Poppler) is missing."
                return new_state
            except Exception as pdf_e:
                logger.exception(f"[AGENT OCR-OpenAI] Error converting PDF to images: {pdf_e}")
                new_state["ocr_error_message"] = f"Error during PDF to image conversion: {str(pdf_e)[:100]}..."
                return new_state
        else:
            new_state["ocr_error_message"] = f"Unsupported file type ({file_extension}). Please upload an image or PDF."
            return new_state

        if not image_messages_content:
            new_state["ocr_error_message"] = "Could not prepare document content for OCR analysis."
            return new_state

        ocr_prompt_text = "Extract all text content from the provided image(s). If it's a medical document or lab report, accurately transcribe all text, values, units, and medical terms. For multi-page documents (multiple images), combine extracted text sequentially."
        image_messages_content.insert(0, {"type": "text", "text": ocr_prompt_text})

        response = llm_openai_vision_ocr.invoke([{"role": "user", "content": image_messages_content}])
        extracted_text = response.content

        if extracted_text and extracted_text.strip():
            new_state["ocr_extracted_text"] = extracted_text.strip()
            logger.info(
                f"[AGENT OCR-OpenAI] Text extracted (first 100 chars): {new_state['ocr_extracted_text'][:100]}...")
        else:
            new_state["ocr_error_message"] = "No text could be extracted from the document by OCR."
    except MedicalDocument.DoesNotExist:
        logger.error(f"[AGENT OCR-OpenAI] MedicalDocument with ID {medical_doc_id} not found.")
        new_state["ocr_error_message"] = "Document not found for OCR."
    except FileNotFoundError:
        logger.error(f"[AGENT OCR-OpenAI] File not found on server for MedicalDocument ID {medical_doc_id}.")
        new_state["ocr_error_message"] = "Document file is missing on server."
    except Exception as e:
        logger.exception(f"[AGENT OCR-OpenAI] Unexpected error during OCR: {e}")
        new_state["ocr_error_message"] = f"An unexpected OCR error occurred: {str(e)[:100]}..."
    logger.debug("[AGENT OCR-OpenAI] === EXITING ocr_agent_openai ===")
    return new_state


# Reasoning Agent (reasoning_agent - RAG logic uses the globally configured embeddings_model_for_rag)
# ... (Insert the full code for reasoning_agent here, it doesn't change for this fix, as it uses the global `embeddings_model_for_rag`)
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
    raw_chat_history = state.get("chat_history", [])
    formatted_history = format_chat_history_for_llm(raw_chat_history, max_messages=6)

    if not llm_openai_reasoning_revising:
        logger.error("[AGENT REASONING-RAG] Reasoning LLM not initialized.")
        new_state["reasoning_error_message"] = "Reasoning service is unavailable."
        return new_state

    query_parts_for_rag = []
    if user_input_text and user_input_text.strip():
        query_parts_for_rag.append(user_input_text.strip())
    if ocr_text and ocr_text.strip() and "Error:" not in ocr_text and "No text could be extracted" not in ocr_text:
        query_parts_for_rag.append(ocr_text.strip())

    consolidated_query_for_rag = "\n".join(query_parts_for_rag).strip()
    new_state["query_for_rag"] = consolidated_query_for_rag
    query_for_embedding_search = consolidated_query_for_rag

    is_persian_heuristic = any(u'\u0600' <= char <= u'\u06FF' for char in consolidated_query_for_rag)
    if consolidated_query_for_rag and is_persian_heuristic:
        logger.info("[AGENT REASONING-RAG] Query appears Persian. Translating for RAG.")
        try:
            translate_prompt_msgs = [
                SystemMessage(
                    content="Translate the following text accurately to English. Output ONLY the translated English text."),
                HumanMessage(content=consolidated_query_for_rag)
            ]
            # Use the main reasoning/revising LLM for translation
            if llm_openai_reasoning_revising:
                translation_response = llm_openai_reasoning_revising.invoke(translate_prompt_msgs)
                translated_text = translation_response.content.strip()
                if translated_text:
                    query_for_embedding_search = translated_text
                    new_state["translated_query_for_rag"] = query_for_embedding_search
                    logger.info(
                        f"[AGENT REASONING-RAG] Translated query for RAG: {query_for_embedding_search[:100]}...")
                else:
                    logger.warning("[AGENT REASONING-RAG] Translation empty. Using original for RAG.")
            else:
                logger.error("[AGENT REASONING-RAG] Translation LLM not available. Using original query for RAG.")
        except Exception as trans_e:
            logger.exception(f"[AGENT REASONING-RAG] Error translating query: {trans_e}. Using original for RAG.")

    retrieved_docs_content_for_prompt = []
    if query_for_embedding_search and vector_store and embeddings_model_for_rag:  # Check global vars
        logger.info(f"[AGENT REASONING-RAG] Performing RAG search. Query: '{query_for_embedding_search[:100]}...'")
        try:
            query_vector = embeddings_model_for_rag.embed_query(query_for_embedding_search)  # Uses global model
            logger.info(f"[AGENT REASONING-RAG] Generated query vector dimension (d_query): {len(query_vector)}")

            logger.debug("[AGENT REASONING-RAG] --- Calling vector_store.similarity_search ---")
            retrieved_documents = vector_store.similarity_search(query_for_embedding_search, k=3)
            logger.debug(f"[AGENT REASONING-RAG] --- vector_store.similarity_search call finished ---")
            logger.info(f"[AGENT REASONING-RAG] Raw result from similarity_search: {retrieved_documents}")

            if retrieved_documents:
                logger.info(f"[AGENT REASONING-RAG] Successfully retrieved {len(retrieved_documents)} documents.")
                new_state["retrieved_rag_documents"] = [{"page_content": doc.page_content, "metadata": doc.metadata} for
                                                        doc in retrieved_documents]
                for i, doc in enumerate(retrieved_documents):
                    retrieved_docs_content_for_prompt.append(
                        f"Retrieved Document {i + 1} (source: {doc.metadata.get('source', 'N/A')}):\n{doc.page_content}")
                logger.info(f"[AGENT REASONING-RAG] Processed {len(retrieved_documents)} docs for context.")
            else:
                logger.info("[AGENT REASONING-RAG] No relevant documents were returned by similarity_search.")
                new_state["retrieved_rag_documents"] = []
        except AssertionError as ae:
            query_dim = "N/A";
            index_dim = "N/A"
            try:
                if embeddings_model_for_rag: query_dim = len(
                    embeddings_model_for_rag.embed_query(query_for_embedding_search))
                if vector_store and vector_store.index: index_dim = vector_store.index.d
            except:
                pass
            logger.error(f"AssertionError during search! d_query: {query_dim}, d_index: {index_dim}. Error: {ae}")
            new_state["reasoning_error_message"] = f"RAG dimension mismatch (query: {query_dim}, index: {index_dim})."
            new_state["retrieved_rag_documents"] = []
        except Exception as rag_e:
            logger.exception(f"[AGENT REASONING-RAG] An exception occurred during RAG search: {rag_e}")
            new_state["reasoning_error_message"] = f"Error during knowledge base lookup: {str(rag_e)[:50]}"
            new_state["retrieved_rag_documents"] = []
    else:
        if not vector_store:
            logger.warning("[AGENT REASONING-RAG] RAG search skipped: vector_store is None.")
        elif not embeddings_model_for_rag:
            logger.warning("[AGENT REASONING-RAG] RAG search skipped: embeddings_model_for_rag is None.")
        elif not query_for_embedding_search:
            logger.warning("[AGENT REASONING-RAG] RAG search skipped: query_for_embedding_search is empty.")
        new_state["retrieved_rag_documents"] = []

    # ... (rest of reasoning_agent logic to prepare prompt and call LLM - remains the same) ...
    reasoning_llm_input_parts = []
    current_query_info_for_llm = []
    if user_input_text and user_input_text.strip():
        current_query_info_for_llm.append(f"User's original latest query/message: \"{user_input_text.strip()}\"")
    ocr_error_from_state = state.get("ocr_error_message")
    if ocr_error_from_state:
        current_query_info_for_llm.append(f"Note on document processing for the user's query: {ocr_error_from_state}")
    elif ocr_text and ocr_text.strip():
        current_query_info_for_llm.append(
            f"Extracted text from user's uploaded document:\n--- DOCUMENT START ---\n{ocr_text.strip()}\n--- DOCUMENT END ---")

    if not current_query_info_for_llm and not formatted_history and not retrieved_docs_content_for_prompt:
        new_state[
            "initial_reasoning_output"] = "No specific information was provided or retrieved for detailed analysis."
        return new_state

    if current_query_info_for_llm:
        reasoning_llm_input_parts.append(
            "Current User Input & Document Scan Details (Original Language if applicable):")
        reasoning_llm_input_parts.extend(current_query_info_for_llm)
        reasoning_llm_input_parts.append("\n---")

    if retrieved_docs_content_for_prompt:
        reasoning_llm_input_parts.append("Relevant Information from Medical Knowledge Base (in English):")
        reasoning_llm_input_parts.extend(retrieved_docs_content_for_prompt)
        reasoning_llm_input_parts.append("\n---")
    else:
        reasoning_llm_input_parts.append(
            "No specific information was retrieved from the medical knowledge base for this query.")
        reasoning_llm_input_parts.append("\n---")

    system_prompt_reasoning_rag = """
You are an AI analytical engine. Your task is to deeply analyze the provided medical-related information. This includes:
1. The user's latest query/uploaded document (in its original language).
2. Relevant preceding conversation history (if provided).
3. Relevant excerpts from a medical knowledge base (provided in English, if any were found).
Your goal is to:
- Synthesize all this information.
- Identify key medical facts, symptoms, test results, or questions.
- If knowledge base excerpts are provided, use them to inform your analysis.
- Identify potential patterns or areas that might warrant further attention (without making a diagnosis).
- Structure your analysis clearly. It is an internal monologue or a detailed breakdown for another AI (a revisor) to use.
- Your analysis output should primarily be in English to facilitate the next review step.
Do NOT provide medical advice, diagnoses, or treatment plans in this output.
If there were errors in document processing or RAG retrieval, acknowledge them.
Output your detailed analysis.
"""
    messages_for_prompt = [SystemMessage(content=system_prompt_reasoning_rag)]
    messages_for_prompt.extend(formatted_history)

    final_reasoning_context_for_llm = "\n".join(reasoning_llm_input_parts)
    if final_reasoning_context_for_llm:
        messages_for_prompt.append(HumanMessage(content=final_reasoning_context_for_llm))
    elif not formatted_history:
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


# Revisor Agent (revisor_agent - remains the same as your last version)
# ... (Insert the full code for revisor_agent here, it doesn't change for this RAG embedding model fix)
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
    retrieved_rag_docs_summary = state.get("retrieved_rag_documents")
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

    if retrieved_rag_docs_summary and len(retrieved_rag_docs_summary) > 0:
        revisor_input_context_parts.append(
            f"- The system retrieved {len(retrieved_rag_docs_summary)} relevant snippets from its knowledge base which informed the internal analysis.")
    elif state.get("query_for_rag"):
        revisor_input_context_parts.append(
            "- No specific snippets were retrieved from the knowledge base for the latest query, or RAG was skipped due to an issue.")

    if reasoning_error:
        revisor_input_context_parts.append(f"- An issue occurred during internal analysis: {reasoning_error}")
        revisor_input_context_parts.append(
            f"- Preliminary internal analysis was: {initial_analysis if initial_analysis else 'Not available due to error.'}")
    elif initial_analysis:
        revisor_input_context_parts.append(
            f"- Internal AI analysis notes (use these to craft the user response, but rephrase into natural language):\n--- INTERNAL ANALYSIS ---\n{initial_analysis}\n--- END INTERNAL ANALYSIS ---")
    else:
        revisor_input_context_parts.append(
            "- No detailed internal analysis was performed for this turn (e.g., awaiting more specific input or due to prior errors).")

    final_input_for_llm_prompt = "\n".join(revisor_input_context_parts)
    last_user_message_content = ""
    if raw_chat_history:
        last_msg_obj = raw_chat_history[-1]
        if last_msg_obj.get("sender") == "user":
            last_user_message_content = last_msg_obj.get("message", "")
    elif user_input_text:
        last_user_message_content = user_input_text

    system_prompt_revisor_and_responder = f"""
You are "MediCare Guide", a specialized AI medical assistant. Your primary purpose is to provide guidance, information, and support related to health and medical topics.
You will be given the recent conversation history (if any) and a summary of the current interaction (user's latest message, document info, and internal analysis notes which may be based on a knowledge base).
Based on ALL this information, generate a helpful, empathetic, and clear response FOR THE USER.

**LANGUAGE DETERMINATION AND EXECUTION (VERY IMPORTANT):**
1.  **Assess User's Language:** Carefully examine the user's LATEST message (content hinted below, and also as the last message in the chat history). Also, check if Persian was used or requested in recent chat history.
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

User's LATEST message content for your immediate attention on language (use this as a strong hint): "{last_user_message_content}"
---
Now, considering all the above, and the following detailed context for the current turn, generate your response to the user:
"""  # End of system prompt

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
workflow.add_node("reasoning_node", reasoning_agent)
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
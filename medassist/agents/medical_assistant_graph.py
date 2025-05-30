# DoctorAssistant/medassist/agents/medical_assistant_graph.py

import os
import base64
import logging
import json  # For loading doctors.json
from typing import TypedDict, Optional, List, Dict, Any

from django.conf import settings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from medassist.models import MedicalDocument

# from DoctorAssistant.settings import OPENAI_API_KEY # Prefer django.conf.settings

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

# --- RAG Configuration (Using HuggingFace Embeddings - as per your working version) ---
RAG_INDEX_FOLDER_PATH = os.path.join(settings.BASE_DIR, "rag_data")
RAG_INDEX_NAME = "index"
vector_store: Optional[FAISS] = None
embeddings_model_for_rag: Optional[HuggingFaceEmbeddings] = None
logger.info("--- Starting RAG Configuration ---")
if FAISS_HF_AVAILABLE:
    try:
        hf_model_name = "sentence-transformers/all-mpnet-base-v2"
        hf_model_kwargs = {"device": "cpu"}
        embeddings_model_for_rag = HuggingFaceEmbeddings(model_name=hf_model_name, model_kwargs=hf_model_kwargs)
        logger.info(f"Initialized HuggingFace embedding model for RAG: '{hf_model_name}'")
        faiss_file_path = os.path.join(RAG_INDEX_FOLDER_PATH, f"{RAG_INDEX_NAME}.faiss")
        pkl_file_path = os.path.join(RAG_INDEX_FOLDER_PATH, f"{RAG_INDEX_NAME}.pkl")
        if os.path.exists(faiss_file_path) and os.path.exists(pkl_file_path):
            vector_store = FAISS.load_local(
                folder_path=RAG_INDEX_FOLDER_PATH, embeddings=embeddings_model_for_rag,
                index_name=RAG_INDEX_NAME, allow_dangerous_deserialization=True
            )
            logger.info("FAISS vector store loaded successfully from disk.")
            if hasattr(vector_store, 'index') and vector_store.index:
                logger.info(f"IMPORTANT - Loaded FAISS index dimension (d_index / self.d): {vector_store.index.d}")
        else:
            logger.error(f"RAG index files not found in '{RAG_INDEX_FOLDER_PATH}'. RAG disabled.")
            vector_store = None
    except Exception as e:
        logger.exception(f"CRITICAL ERROR during RAG setup: {e}")
        vector_store = None
else:
    logger.warning("RAG features disabled due to missing FAISS/HuggingFace prerequisites.")
    vector_store = None
logger.info("--- Finished RAG Configuration ---")

# --- Load Doctor Data ---
DOCTORS_JSON_PATH = os.path.join(settings.BASE_DIR, "doctors_list.json")  # Ensure this path is correct
doctors_data: List[Dict[str, Any]] = []
logger.info("--- Starting Doctor Data Configuration ---")
try:
    if os.path.exists(DOCTORS_JSON_PATH):
        with open(DOCTORS_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "doctors_list" in data and isinstance(data["doctors_list"], list):
                doctors_data = data["doctors_list"]
                logger.info(f"Successfully loaded {len(doctors_data)} doctor profiles from '{DOCTORS_JSON_PATH}'.")
            else:
                logger.error(
                    f"'doctors_list' key not found or not a list in '{DOCTORS_JSON_PATH}'. No doctor data loaded.")
    else:
        logger.error(f"Doctor data file not found: '{DOCTORS_JSON_PATH}'. No doctor data loaded.")
except Exception as e:
    logger.exception(f"Error loading doctors_list.json: {e}")
logger.info("--- Finished Doctor Data Configuration ---")

# --- Configure LLMs (Your existing 2 LLM setup) ---
llm_openai_reasoning_revising: Optional[
    ChatOpenAI] = None  # Will be used for Reasoning, Revising, and Keyword Extraction
llm_openai_vision_ocr: Optional[ChatOpenAI] = None  # For OCR

if settings.OPENAI_API_KEY:
    try:
        llm_openai_reasoning_revising = ChatOpenAI(
            model="gpt-4o",  # As per your current code
            temperature=0.5,
            api_key=settings.OPENAI_API_KEY,
            max_tokens=1500
        )
        logger.info(
            f"OpenAI LLM for Reasoning/Revising/Keywords ({llm_openai_reasoning_revising.model_name}) initialized.")

        llm_openai_vision_ocr = ChatOpenAI(
            model="gpt-4o",  # As per your current code
            temperature=0.2,
            api_key=settings.OPENAI_API_KEY,
            max_tokens=2048
        )
        logger.info(f"OpenAI LLM for Vision/OCR ({llm_openai_vision_ocr.model_name}) initialized.")
    except Exception as e:
        logger.error(f"Error initializing OpenAI LLMs: {e}")
else:
    logger.error("OPENAI_API_KEY is not set. Cannot initialize main Chat LLMs.")


# --- LangGraph State Definition ---
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
    # New state for doctor recommendation
    identified_health_keywords: Optional[List[str]]  # Keywords extracted for doctor search
    # recommended_doctors_info is not stored in state, it's integrated into final_llm_response


# --- Helper function for chat history ---
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


# --- Doctor Search Helper Functions ---
def extract_keywords_for_doctor_search(
    text_analysis: Optional[str],
    user_query: Optional[str],
    ocr_content: Optional[str],  # Could be JSON string
    llm: Optional[ChatOpenAI]    # This is the 4th argument
) -> List[str]:
    # This function will now use the passed 'llm' instance
    if not llm:
        logger.warning("[DOCTOR KEYWORDS] LLM for keyword extraction not provided or not available.")
        # Basic fallback if LLM is not available (less accurate)
        combined_text = (text_analysis or "") + " " + (user_query or "") + " " + (ocr_content or "")
        # Simple split, you might want to improve this fallback
        return [word for word in combined_text.lower().split() if len(word) > 3][:5]

    prompt_context = f"""
    Based on the following medical information, identify key medical terms, conditions, symptoms, or relevant medical specialties.
    This will be used to find a suitable doctor.
    Focus on specific medical nouns and specialties.
    If OCR content is JSON, interpret the key findings.

    User Query: "{user_query if user_query else 'N/A'}"
    OCR Content (first 500 chars, may be JSON): "{ocr_content[:500] if ocr_content else 'N/A'}..." 
    Medical Analysis Provided: "{text_analysis if text_analysis else 'N/A'}"

    Extract up to 5-7 distinct keywords or short phrases relevant for a doctor search. Output ONLY a comma-separated list.
    Example: cardiology, chest pain, arrhythmia, skin rash, pediatric dermatology
    Keywords:
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt_context)])
        keywords_str = response.content.strip()
        # Robustly split and clean keywords
        keywords_list = [kw.strip().lower() for kw in keywords_str.split(',') if kw.strip() and len(kw.strip()) > 2]
        logger.info(f"[DOCTOR KEYWORDS] Extracted: {keywords_list}")
        return list(set(keywords_list[:7]))  # Return unique keywords, max 7
    except Exception as e:
        logger.exception(f"[DOCTOR KEYWORDS] Error extracting keywords using LLM: {e}")
        return [] # Return empty list on error


def search_doctors_by_keywords(keywords: List[str]) -> List[Dict[str, Any]]:
    global doctors_data
    if not keywords or not doctors_data:
        return []

    logger.debug(f"[DOCTOR SEARCH] Searching with keywords: {keywords}")
    matched_doctors = []
    normalized_keywords = [kw.lower().strip() for kw in keywords if kw]

    for doctor in doctors_data:
        if not doctor.get("additional_information", {}).get("is_active", False):
            continue

        searchable_fields_raw = []  # Store raw values first

        # Safely get personal information fields
        personal_info = doctor.get("personal_information", {})
        searchable_fields_raw.append(personal_info.get("full_name_with_prefix"))
        searchable_fields_raw.append(personal_info.get("sub_specialty"))

        # Safely get specialization and services fields
        spec_services = doctor.get("specialization_and_services", {})
        searchable_fields_raw.append(spec_services.get("main_specialty"))
        searchable_fields_raw.extend(spec_services.get("list_of_services_treatments", []))
        searchable_fields_raw.extend(spec_services.get("proficient_in_treating_specific_diseases", []))

        # Safely get additional information keywords
        additional_info = doctor.get("additional_information", {})
        searchable_fields_raw.extend(additional_info.get("keywords_for_search", []))

        # Convert to lowercase only if the field is a string, filter out None values
        searchable_fields_lower = []
        for field_value in searchable_fields_raw:
            if isinstance(field_value, str):
                searchable_fields_lower.append(field_value.lower())
            # If it's a list (like services), it's already handled by extend,
            # but individual items in those lists also need to be strings.
            # The list appends above are fine as they extend with lists of strings or empty lists.

        doctor_text_corpus = " ".join(filter(None,
                                             searchable_fields_lower))  # filter(None, ...) will remove any None values that might have slipped through if not strings

        score = 0
        for kw in normalized_keywords:
            if kw in doctor_text_corpus:  # Search in the combined lowercase string
                score += 1

        if score > 0:
            doctor_info_with_score = doctor.copy()
            doctor_info_with_score['_match_score'] = score
            matched_doctors.append(doctor_info_with_score)

    matched_doctors.sort(key=lambda x: x['_match_score'], reverse=True)
    logger.info(f"[DOCTOR SEARCH] Found {len(matched_doctors)} potential doctor matches.")
    return matched_doctors[:2]


# --- Agent Nodes ---

# OCR Agent (Uses llm_openai_vision_ocr - gpt-4o)
def ocr_agent_openai(state: AgentState) -> AgentState:
    logger.debug("[AGENT OCR] === ENTERING ===")  # Simplified log prefix
    new_state = state.copy();
    new_state["ocr_extracted_text"] = None;
    new_state["ocr_error_message"] = None
    medical_doc_id = state.get("medical_document_id")

    # Use the globally defined llm_openai_vision_ocr
    if not llm_openai_vision_ocr:
        new_state["ocr_error_message"] = "OCR service unavailable.";
        return new_state
    if not medical_doc_id: logger.debug("[AGENT OCR] No doc ID. Skipping."); return new_state
    try:
        doc = MedicalDocument.objects.get(id=medical_doc_id)
        if not doc.file or not doc.file.name: new_state[
            "ocr_error_message"] = "Doc record found, file missing."; return new_state
        file_path = doc.file.path;
        file_name = os.path.basename(doc.file.name);
        file_extension = os.path.splitext(file_name)[1].lower()
        image_messages_content = []
        if file_extension in [".jpg", ".jpeg", ".png", ".webp", ".gif"]:
            with open(file_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            image_messages_content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/{file_extension[1:]};base64,{base64_image}"}})
        elif file_extension == ".pdf":
            try:
                from pdf2image import convert_from_path
                images_from_pdf = convert_from_path(file_path, dpi=200, fmt='png', thread_count=2)
                if not images_from_pdf: new_state[
                    "ocr_error_message"] = "Could not extract pages from PDF."; return new_state
                for i, page_image in enumerate(images_from_pdf):
                    if i >= 3: logger.info(f"OCR: Processing first 3 PDF pages of {file_name}."); break
                    import io;
                    buffered = io.BytesIO();
                    page_image.save(buffered, format="PNG")
                    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    image_messages_content.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})
                if not image_messages_content: new_state["ocr_error_message"] = "No images from PDF."; return new_state
            except ImportError:
                new_state["ocr_error_message"] = "PDF processing lib missing (pdf2image/Poppler)."; return new_state
            except Exception as pdf_e:
                new_state["ocr_error_message"] = f"PDF conversion error: {str(pdf_e)[:100]}..."; return new_state
        else:
            new_state["ocr_error_message"] = f"Unsupported file type ({file_extension})."; return new_state
        if not image_messages_content: new_state[
            "ocr_error_message"] = "Could not prepare document for OCR."; return new_state

        # Using the specific OCR prompt you provided earlier
        ocr_system_prompt_text = """You are an expert OCR system specialized in extracting medical lab test results. 
Your task is to accurately read the provided image(s) of a lab test report and for each individual lab parameter, extract its Test name, its corresponding Result (numerical or textual), its Unit, and its Reference Range. 
Organize the extracted data as a JSON object, where the top-level key is 'Tests' and its value is an array. Each element in the 'Tests' array should be an object. 
Include the 'Test' key and its value for every identified test. 
For 'Result', 'Unit', and 'Reference_Range', only include the key-value pair if a clear value is successfully extracted. 
If a 'Test' name or its 'Result' cannot be identified for an entry, omit the entire object for that entry from the 'Tests' array to ensure data completeness.
Output ONLY the JSON object. Do not include any other explanatory text or markdown formatting like ```json ... ```."""

        final_ocr_prompt_content = [{"type": "text", "text": ocr_system_prompt_text}]
        final_ocr_prompt_content.extend(image_messages_content)

        response = llm_openai_vision_ocr.invoke([HumanMessage(content=final_ocr_prompt_content)])
        extracted_content_str = response.content.strip()
        try:
            json.loads(extracted_content_str)
            new_state["ocr_extracted_text"] = extracted_content_str
            logger.info(f"[AGENT OCR] OCR output is valid JSON (first 100 chars): {extracted_content_str[:100]}...")
        except json.JSONDecodeError:
            logger.warning(
                f"[AGENT OCR] OCR output not valid JSON, storing as raw text: {extracted_content_str[:100]}...")
            new_state["ocr_extracted_text"] = extracted_content_str
            if not extracted_content_str: new_state["ocr_error_message"] = "No text/data extracted by OCR."
        if not new_state["ocr_extracted_text"] and not new_state["ocr_error_message"]:
            new_state["ocr_error_message"] = "OCR resulted in empty content."
    except Exception as e:
        logger.exception(f"[AGENT OCR] Error: {e}"); new_state["ocr_error_message"] = f"OCR error: {str(e)[:100]}..."
    logger.debug("[AGENT OCR] === EXITING ===")
    return new_state


# Reasoning Agent (Uses llm_openai_reasoning_revising - gpt-4o, as per your current code)
def reasoning_agent(state: AgentState) -> AgentState:
    logger.debug("[AGENT REASONING-RAG] === ENTERING ===")
    new_state = state.copy()
    # ... (Reset fields as in your current reasoning_agent)
    new_state["initial_reasoning_output"] = None;
    new_state["reasoning_error_message"] = None
    new_state["query_for_rag"] = None;
    new_state["translated_query_for_rag"] = None
    new_state["retrieved_rag_documents"] = []

    user_input_text = state.get("current_user_input_text")
    ocr_text = state.get("ocr_extracted_text")  # This could be a JSON string
    raw_chat_history = state.get("chat_history", [])
    formatted_history = format_chat_history_for_llm(raw_chat_history, max_messages=6)

    if not llm_openai_reasoning_revising:  # This is the gpt-4o instance in your current code
        logger.error("[AGENT REASONING-RAG] Reasoning LLM (llm_openai_reasoning_revising) not initialized.")
        new_state["reasoning_error_message"] = "Reasoning service unavailable.";
        return new_state

    # --- Query Consolidation & Translation for RAG (as in your current code) ---
    query_parts_for_rag = []
    if user_input_text and user_input_text.strip(): query_parts_for_rag.append(user_input_text.strip())
    if ocr_text and ocr_text.strip() and not state.get("ocr_error_message"):
        query_parts_for_rag.append(f"OCR'd Content (may be JSON):\n{ocr_text.strip()}")
    elif state.get("ocr_error_message"):
        query_parts_for_rag.append(f"Note on document processing: {state.get('ocr_error_message')}")
    consolidated_query_for_rag = "\n".join(query_parts_for_rag).strip()
    if not consolidated_query_for_rag and not formatted_history:
        new_state["initial_reasoning_output"] = "No information provided for reasoning.";
        return new_state
    new_state["query_for_rag"] = consolidated_query_for_rag
    query_for_embedding_search = consolidated_query_for_rag
    is_persian_heuristic = any(u'\u0600' <= char <= u'\u06FF' for char in consolidated_query_for_rag)
    if consolidated_query_for_rag and is_persian_heuristic:
        try:
            translate_prompt_msgs = [SystemMessage(content="Translate to English. Output ONLY translated text."),
                                     HumanMessage(content=consolidated_query_for_rag)]
            if llm_openai_reasoning_revising:  # Using the gpt-4o for translation
                translation_response = llm_openai_reasoning_revising.invoke(translate_prompt_msgs)
                translated_text = translation_response.content.strip()
                if translated_text: query_for_embedding_search = translated_text; new_state[
                    "translated_query_for_rag"] = query_for_embedding_search
            else:
                logger.error("[AGENT REASONING-RAG] Translation LLM not available.")
        except Exception as trans_e:
            logger.exception(f"[AGENT REASONING-RAG] Error translating: {trans_e}.")

    # --- RAG Search (as in your current code) ---
    retrieved_docs_content_for_prompt = []
    if query_for_embedding_search and vector_store and embeddings_model_for_rag:
        try:
            # query_vector = embeddings_model_for_rag.embed_query(query_for_embedding_search) # Already logged if needed
            retrieved_documents = vector_store.similarity_search(query_for_embedding_search, k=3)
            if retrieved_documents:
                new_state["retrieved_rag_documents"] = [{"page_content": doc.page_content, "metadata": doc.metadata} for
                                                        doc in retrieved_documents]
                for i, doc in enumerate(retrieved_documents): retrieved_docs_content_for_prompt.append(
                    f"Retrieved Doc {i + 1}:\n{doc.page_content}")
        except Exception as rag_e:
            logger.exception(f"[AGENT REASONING-RAG] RAG search error: {rag_e}")
    new_state["retrieved_rag_documents"] = new_state.get("retrieved_rag_documents", [])

    # --- Prepare context for reasoning LLM (as in your current code) ---
    reasoning_llm_input_parts = []
    # ... (Construct reasoning_llm_input_parts including current query, OCR JSON string, RAG docs) ...
    current_query_info_for_llm = []
    if user_input_text and user_input_text.strip(): current_query_info_for_llm.append(
        f"User's query: \"{user_input_text.strip()}\"")
    ocr_error_from_state = state.get("ocr_error_message")
    if ocr_error_from_state:
        current_query_info_for_llm.append(f"Document note: {ocr_error_from_state}")
    elif ocr_text and ocr_text.strip():
        current_query_info_for_llm.append(
            f"Extracted document content (this might be structured JSON from lab results - interpret it accordingly):\n{ocr_text.strip()}")
    if not current_query_info_for_llm and not formatted_history and not retrieved_docs_content_for_prompt:
        new_state["initial_reasoning_output"] = "No information for analysis.";
        return new_state  #
    if current_query_info_for_llm:
        reasoning_llm_input_parts.append("Current User Input & Document Details:");
        reasoning_llm_input_parts.extend(current_query_info_for_llm);
        reasoning_llm_input_parts.append("\n---")
    if retrieved_docs_content_for_prompt:
        reasoning_llm_input_parts.append("Relevant Info from Knowledge Base (English):");
        reasoning_llm_input_parts.extend(retrieved_docs_content_for_prompt);
        reasoning_llm_input_parts.append("\n---")
    else:
        reasoning_llm_input_parts.append("No info retrieved from knowledge base.\n---")

    system_prompt_reasoning_rag = """
You are an AI analytical engine. Your task is to deeply analyze provided medical-related information:
1. User's latest query/uploaded document (OCR output might be a JSON string of test results - you MUST interpret this JSON if present).
2. Conversation history.
3. Relevant knowledge base excerpts (if any, in English).
Goal: Synthesize all information. Identify key medical facts, symptoms, lab results (paying close attention to values, units, reference ranges if OCR provided JSON), or questions. Use knowledge base excerpts. Identify potential patterns, anomalies, or areas that might warrant further attention (DO NOT DIAGNOSE).
Output: Structured, detailed analysis (internal monologue/breakdown) for another AI (a revisor) to use. Output in English. Acknowledge processing errors.
"""
    messages_for_prompt = [SystemMessage(content=system_prompt_reasoning_rag)]
    messages_for_prompt.extend(formatted_history)
    final_reasoning_context_for_llm = "\n".join(reasoning_llm_input_parts)
    if final_reasoning_context_for_llm:
        messages_for_prompt.append(HumanMessage(content=final_reasoning_context_for_llm))
    elif not formatted_history:
        new_state["initial_reasoning_output"] = "No info to analyze."; return new_state

    logger.debug(
        f"[AGENT REASONING-RAG] Final context for reasoning (using llm_openai_reasoning_revising). Input: {messages_for_prompt[-1].content[:150]}...")
    try:
        response_obj = llm_openai_reasoning_revising.invoke(messages_for_prompt)  # Uses the gpt-4o instance
        new_state["initial_reasoning_output"] = response_obj.content
    except Exception as e:
        logger.exception(f"[AGENT REASONING-RAG] Error with llm_openai_reasoning_revising: {e}")
        new_state["reasoning_error_message"] = f"Analysis error: {str(e)[:100]}..."
        new_state["initial_reasoning_output"] = "Failed detailed analysis."
    logger.debug("[AGENT REASONING-RAG] === EXITING ===")
    return new_state


# Revisor Agent (Uses llm_openai_reasoning_revising - gpt-4o, as per your current code + Doctor Rec)
def revisor_agent(state: AgentState) -> AgentState:
    logger.debug("[AGENT REVISOR] === ENTERING ===")
    new_state = state.copy()
    new_state["final_llm_response"] = None;
    new_state["revisor_error_message"] = None
    new_state["identified_health_keywords"] = None

    user_input_text = state.get("current_user_input_text")
    ocr_text = state.get("ocr_extracted_text")
    ocr_error = state.get("ocr_error_message")
    initial_analysis = state.get("initial_reasoning_output")
    reasoning_error = state.get("reasoning_error_message")
    raw_chat_history = state.get("chat_history", [])
    formatted_history_for_revisor = format_chat_history_for_llm(raw_chat_history, max_messages=8)

    if not llm_openai_reasoning_revising:  # This is the gpt-4o instance in your current code
        logger.error("[AGENT REVISOR] Revising LLM (llm_openai_reasoning_revising) not initialized.")
        new_state["final_llm_response"] = "Error: AI response finalization service unavailable.";
        return new_state

    # --- Doctor Recommendation Logic ---
    extracted_keywords_for_doc_search = []
    doctor_recommendation_string_for_prompt = ""

    trigger_doctor_search = False
    if initial_analysis and not reasoning_error:
        # Heuristic to decide if doctor search is relevant
        analysis_lower = initial_analysis.lower()
        if any(kw in analysis_lower for kw in
               ["concern", "issue", "abnormal", "condition", "symptom", "critical", "serious", "see doctor",
                "consultation", "further evaluation", "low", "high", "elevated", "reduced", "requires attention"]):
            trigger_doctor_search = True
            logger.info(
                f"[AGENT REVISOR] Heuristic: Analysis suggests a health issue. Triggering doctor search based on: '{analysis_lower[:100]}...'")

    if trigger_doctor_search:
        logger.info("[AGENT REVISOR] Attempting to extract keywords for doctor search.")
        # Use the existing llm_openai_reasoning_revising (gpt-4o) for keyword extraction
        extracted_keywords_for_doc_search = extract_keywords_for_doctor_search(
            initial_analysis, user_input_text, ocr_text, llm_openai_reasoning_revising
        )
        new_state["identified_health_keywords"] = extracted_keywords_for_doc_search

        if extracted_keywords_for_doc_search and doctors_data:
            matched_doctors = search_doctors_by_keywords(extracted_keywords_for_doc_search)
            if matched_doctors:
                recommendations = [
                    "Additionally, based on the discussion, here are some specialists who might be relevant (this is not an endorsement, and you should verify their suitability):"]
                for i, doc_profile in enumerate(matched_doctors):  # matched_doctors already returns top 2
                    doc_info = f"{i + 1}. {doc_profile.get('personal_information', {}).get('full_name_with_prefix', 'N/A')}"
                    doc_info += f" - Specialty: {doc_profile.get('specialization_and_services', {}).get('main_specialty', 'N/A')}"
                    sub_spec = doc_profile.get('personal_information', {}).get('sub_specialty')
                    if sub_spec: doc_info += f" ({sub_spec})"
                    doc_info += f" in {doc_profile.get('contact_and_location', {}).get('city', 'N/A')}."
                    working_hours = doc_profile.get('contact_and_location', {}).get('working_days_hours')
                    if working_hours: doc_info += f" Availability: {working_hours}."
                    # Could add phone or online appointment link if available and desired
                    # online_appt = doc_profile.get('contact_and_location',{}).get('online_appointment_link')
                    # if online_appt: doc_info += f" Appointments: {online_appt}"
                    recommendations.append(doc_info)
                doctor_recommendation_string_for_prompt = "\n".join(recommendations)
                logger.info(
                    f"[AGENT REVISOR] Formatted doctor recommendation for prompt: {doctor_recommendation_string_for_prompt}")
            else:
                logger.info("[AGENT REVISOR] No matching doctors found based on keywords.")
        elif not doctors_data:
            logger.warning("[AGENT REVISOR] No doctor data loaded for recommendations.")

    # --- Prepare context for final response generation by Revisor LLM ---
    revisor_input_context_parts = []
    # ... (Construct revisor_input_context_parts as before, including initial_analysis etc.)
    revisor_input_context_parts.append("Current Interaction Context:")
    if user_input_text and user_input_text.strip(): revisor_input_context_parts.append(
        f"- User's message: \"{user_input_text.strip()}\"")
    if ocr_error:
        revisor_input_context_parts.append(f"- Document note: {ocr_error}")
    elif ocr_text and ocr_text.strip():
        revisor_input_context_parts.append(f"- Document info (summary): \"{ocr_text[:300]}...\"")
    retrieved_rag_docs_summary = state.get("retrieved_rag_documents")
    if retrieved_rag_docs_summary and len(retrieved_rag_docs_summary) > 0:
        revisor_input_context_parts.append(f"- {len(retrieved_rag_docs_summary)} knowledge base snippets considered.")
    elif state.get("query_for_rag"):
        revisor_input_context_parts.append("- No specific knowledge base snippets retrieved or RAG skipped.")
    if reasoning_error:
        revisor_input_context_parts.append(
            f"- Internal analysis issue: {reasoning_error}"); revisor_input_context_parts.append(
            f"- Preliminary analysis: {initial_analysis if initial_analysis else 'N/A.'}")
    elif initial_analysis:
        revisor_input_context_parts.append(
            f"- Internal AI analysis notes (rephrase for user):\n--- ANALYSIS ---\n{initial_analysis}\n--- END ---")
    else:
        revisor_input_context_parts.append("- No detailed internal analysis performed.")

    if doctor_recommendation_string_for_prompt:
        revisor_input_context_parts.append(
            "\n--- POTENTIAL DOCTOR SUGGESTION INFORMATION (If appropriate, naturally weave this into your response towards the end, after the main health discussion and disclaimer. Ensure it fits the conversational flow and language.) ---")
        revisor_input_context_parts.append(doctor_recommendation_string_for_prompt)
        revisor_input_context_parts.append("--- END DOCTOR SUGGESTION INFORMATION ---")

    final_input_for_llm_prompt = "\n".join(revisor_input_context_parts)
    last_user_message_content = ""
    if raw_chat_history and raw_chat_history[-1].get("sender") == "user":
        last_user_message_content = raw_chat_history[-1].get("message", "")
    elif user_input_text:
        last_user_message_content = user_input_text

    system_prompt_revisor_and_responder = f"""
You are "MediCare Guide", a specialized AI medical assistant.
Task: Generate a comprehensive, empathetic, clear health-related response based on user query, history, document data (possibly JSON from OCR), internal analysis (RAG-informed), and POTENTIAL doctor suggestions.

**LANGUAGE (VERY IMPORTANT):**
- If user's LATEST message (hint below) or recent history is Persian, or Persian is explicitly requested, ENTIRE response MUST be in fluent Persian.
- Otherwise, respond in English.
- ALL parts of response (greetings, main content, disclaimers, doctor suggestions, builder info if asked) MUST be in the determined language.

**DOCTOR SUGGESTION INTEGRATION (If applicable AND IF information is provided in context under "POTENTIAL DOCTOR SUGGESTION INFORMATION"):**
- If doctor suggestions are provided AND your health analysis indicates a need for specialist consultation aligning with these suggestions, NATURALLY integrate them.
- Example intro: "Additionally, if you're considering a specialist, here are some doctors who might be relevant..."
- If no suggestions provided in context, or they don't seem relevant, DO NOT mention doctors.

**Pre-defined Phrases (use version matching response language):**
* Builder Info (if asked "who built you"):
    * EN: "I am an AI assistant developed by the StackToServ team, dedicated to building intelligent AI systems."
    * FA: "من یک دستیار هوش مصنوعی هستم که توسط تیم StackToServ توسعه داده شده‌ام؛ تیمی که به ساخت سیستم‌های هوشمند هوش مصنوعی اختصاص دارد."
* Medical Disclaimer (MANDATORY at end of relevant responses):
    * EN: "Please remember, this is AI-generated information, not medical advice. Always consult a doctor for personal health concerns."
    * FA: "لطفا به یاد داشته باشید، این اطلاعات توسط هوش مصنوعی تولید شده و جایگزین توصیه پزشکی حرفه‌ای نیست. همیشه برای هرگونه نگرانی سلامتی یا قبل از تصمیم‌گیری در مورد سلامتی خود با پزشک یا ارائه‌دهنده خدمات بهداشتی واجد شرایط مشورت کنید."

**Response Guidelines (in chosen language):**
1. Acknowledge input. Synthesize info from internal analysis & docs (interpret OCR JSON if present).
2. Handle errors gracefully.
3. Maintain "MediCare Guide" persona: health-focused, professional, empathetic. Redirect off-topic LATEST queries.
4. CRITICAL: No medical advice/diagnosis/prescription. ALWAYS include Medical Disclaimer.
5. Use history for context.

User's LATEST message (for language hint): "{last_user_message_content}"
---
Now, using all context, generate the final response:
"""
    messages_for_llm = [SystemMessage(content=system_prompt_revisor_and_responder)]
    messages_for_llm.extend(formatted_history_for_revisor)
    messages_for_llm.append(HumanMessage(content=final_input_for_llm_prompt))

    logger.debug(
        f"[AGENT REVISOR] Sending context to llm_openai_reasoning_revising. Input: {messages_for_llm[-1].content[:200]}...")
    try:
        response_obj = llm_openai_reasoning_revising.invoke(messages_for_llm)  # Uses the gpt-4o instance
        new_state["final_llm_response"] = response_obj.content
    except Exception as e:
        logger.exception(f"[AGENT REVISOR] Error during llm_openai_reasoning_revising call: {e}")
        new_state["final_llm_response"] = "I apologize, an issue occurred. Please try again."
    logger.debug("[AGENT REVISOR] === EXITING ===")
    return new_state


# --- Define the LangGraph Workflow (No Triage, back to OCR entry) ---
workflow = StateGraph(AgentState)

# Remove triage node for now, directly use OCR -> Reasoning -> Revisor
workflow.add_node("ocr_node", ocr_agent_openai)
workflow.add_node("reasoning_node", reasoning_agent)
workflow.add_node("revisor_node", revisor_agent)  # This includes doctor recommendation logic

workflow.set_entry_point("ocr_node")  # Start with OCR
workflow.add_edge("ocr_node", "reasoning_node")
workflow.add_edge("reasoning_node", "revisor_node")
workflow.add_edge("revisor_node", END)

try:
    compiled_medical_assistant_graph = workflow.compile()
    logger.info(
        "LangGraph with OCR, Reasoning, and Revisor (incl. Doctor Rec) agents compiled successfully (2 LLM config).")
except Exception as e:
    logger.exception("Failed to compile LangGraph with doctor recommendation updates:")
    compiled_medical_assistant_graph = None
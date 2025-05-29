# DoctorAssistant/medassist/agents/medical_assistant_graph.py

import os
import base64 # For encoding images for OpenAI Vision
import logging
from typing import TypedDict, Optional, List, Dict

from django.conf import settings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from medassist.models import MedicalDocument

# For PDF to image conversion (requires pdf2image and poppler)
# from pdf2image import convert_from_bytes # Or convert_from_path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Configure LLMs ---
OPENAI_API_KEY = settings.OPENAI_API_KEY

llm_openai_text_responder = None # For the text agent
llm_openai_vision_ocr = None   # For the OCR agent

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set in Django settings. OpenAI LLMs will not be initialized.")
else:
    try:
        # LLM for general text responses
        llm_openai_text_responder = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=OPENAI_API_KEY)
        logger.info("OpenAI LLM for text responses (gpt-3.5-turbo) initialized successfully.")

        # LLM for Vision/OCR (GPT-4o is a good choice for multimodal tasks)
        # You can also use "gpt-4-turbo" if "gpt-4o" is not preferred or available via your exact setup.
        llm_openai_vision_ocr = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=OPENAI_API_KEY, max_tokens=2048) # Increased max_tokens for potentially long OCR
        logger.info("OpenAI LLM for Vision/OCR (gpt-4o) initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing OpenAI LLMs: {e}")
        if llm_openai_text_responder:
             logger.error("Text responder LLM might have initialized, but vision LLM failed or vice-versa.")


# --- LangGraph State Definition ---
class AgentState(TypedDict):
    user_id: int
    chat_history: List[Dict[str, str]]
    current_user_input_text: Optional[str]
    medical_document_id: Optional[int]
    ocr_extracted_text: Optional[str]
    ocr_error_message: Optional[str]
    final_llm_response: Optional[str]


# --- Agent Nodes ---

def ocr_agent_openai(state: AgentState) -> AgentState:
    logger.debug("[AGENT OCR-OpenAI] === ENTERING ocr_agent_openai ===")
    new_state = state.copy()
    new_state["ocr_extracted_text"] = None
    new_state["ocr_error_message"] = None
    medical_doc_id = state.get("medical_document_id")

    if not llm_openai_vision_ocr:
        logger.error("[AGENT OCR-OpenAI] OpenAI Vision LLM (llm_openai_vision_ocr) not initialized. Skipping OCR.")
        new_state["ocr_error_message"] = "OCR service (OpenAI Vision) is not available."
        return new_state

    if not medical_doc_id:
        logger.debug("[AGENT OCR-OpenAI] No medical_document_id provided. Skipping OCR.")
        return new_state

    try:
        doc = MedicalDocument.objects.get(id=medical_doc_id)
        if not doc.file or not doc.file.name:
            logger.warning(f"[AGENT OCR-OpenAI] MedicalDocument ID {medical_doc_id} has no associated file.")
            new_state["ocr_error_message"] = "Document record found, but the file is missing."
            return new_state

        file_path = doc.file.path
        file_name = os.path.basename(doc.file.name)
        file_extension = os.path.splitext(file_name)[1].lower()

        image_messages_content = [] # List to hold image data for the prompt

        if file_extension in [".jpg", ".jpeg", ".png", ".webp", ".gif"]: # Common image types GPT-4V supports
            logger.info(f"[AGENT OCR-OpenAI] Processing image file: {file_name}")
            with open(file_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            image_messages_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{file_extension[1:]};base64,{base64_image}"}
            })
        elif file_extension == ".pdf":
            logger.info(f"[AGENT OCR-OpenAI] Processing PDF file: {file_name}. Needs page-by-page image conversion.")
            # OpenAI's vision models generally need images. For PDFs, convert pages to images.
            # This requires `pdf2image` and `poppler`.
            # Example (uncomment and install dependencies if you need this):
            try:
                from pdf2image import convert_from_path # Requires poppler
                logger.debug(f"Attempting to convert PDF {file_name} to images...")
                images_from_pdf = convert_from_path(file_path, dpi=200) # dpi can be adjusted
                if not images_from_pdf:
                    logger.warning(f"pdf2image returned no images for {file_name}")
                    new_state["ocr_error_message"] = "Could not extract pages from PDF for OCR."
                    return new_state

                for i, page_image in enumerate(images_from_pdf):
                    if i >= 5: # Limit number of pages to process to control cost/time
                        logger.info(f"Reached page limit (5) for PDF {file_name}. Processing first 5 pages.")
                        break
                    logger.debug(f"Processing page {i+1} of PDF {file_name}")
                    # Convert PIL image to base64
                    import io
                    buffered = io.BytesIO()
                    page_image.save(buffered, format="PNG") # Convert page to PNG
                    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    image_messages_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    })
                if not image_messages_content:
                    new_state["ocr_error_message"] = "No images could be processed from the PDF."
                    return new_state
            except ImportError:
                logger.error("[AGENT OCR-OpenAI] `pdf2image` library not found. Cannot process PDF files. Please install it and poppler.")
                new_state["ocr_error_message"] = "Cannot process PDF for OCR: pdf2image library is missing."
                return new_state
            except Exception as pdf_conv_e:
                logger.exception(f"[AGENT OCR-OpenAI] Error converting PDF page to image: {pdf_conv_e}")
                new_state["ocr_error_message"] = f"Error during PDF to image conversion: {str(pdf_conv_e)[:100]}..."
                return new_state
        else:
            logger.error(f"[AGENT OCR-OpenAI] Unsupported file type: {file_extension} for file {file_name}")
            new_state["ocr_error_message"] = f"Unsupported file type ({file_extension}). Please upload an image (JPG, PNG, WEBP, GIF) or a PDF."
            return new_state

        if not image_messages_content:
             logger.warning("[AGENT OCR-OpenAI] No image content was prepared for OpenAI Vision call.")
             new_state["ocr_error_message"] = "Could not prepare document for OCR analysis."
             return new_state

        # Add the text prompt part
        ocr_prompt_text = "Extract all text content from the provided image(s). If it's a medical document or lab report, try to maintain the structure and accurately transcribe all values, units, and medical terms. If multiple images are provided (e.g., from PDF pages), combine the extracted text sequentially."
        image_messages_content.insert(0, {"type": "text", "text": ocr_prompt_text})

        logger.info(f"[AGENT OCR-OpenAI] Sending {len(image_messages_content)-1} image(s) to OpenAI Vision for OCR.")

        response = llm_openai_vision_ocr.invoke([
            {"role": "user", "content": image_messages_content}
        ])
        extracted_text = response.content # .content should be string

        if extracted_text and extracted_text.strip():
            new_state["ocr_extracted_text"] = extracted_text.strip()
            logger.info(f"[AGENT OCR-OpenAI] Successfully extracted text (first 100 chars): {new_state['ocr_extracted_text'][:100]}...")
        else:
            logger.warning(f"[AGENT OCR-OpenAI] OpenAI Vision extracted no text from: {file_name}")
            new_state["ocr_error_message"] = "No text could be extracted from the document using OpenAI Vision."

    except MedicalDocument.DoesNotExist:
        logger.error(f"[AGENT OCR-OpenAI] MedicalDocument with ID {medical_doc_id} not found.")
        new_state["ocr_error_message"] = "Document not found for OCR."
    except FileNotFoundError:
        logger.error(f"[AGENT OCR-OpenAI] File not found for MedicalDocument ID {medical_doc_id}.")
        new_state["ocr_error_message"] = "Document file is missing on server."
    except Exception as e:
        logger.exception(f"[AGENT OCR-OpenAI] An unexpected error occurred: {e}")
        new_state["ocr_error_message"] = f"An unexpected OCR error occurred: {str(e)[:100]}..."

    logger.debug("[AGENT OCR-OpenAI] === EXITING ocr_agent_openai ===")
    return new_state


def text_responder_agent(state: AgentState) -> AgentState:
    logger.debug("[AGENT TEXT] === ENTERING text_responder_agent ===")
    new_state = state.copy()
    new_state["final_llm_response"] = None
    user_input_text = state.get("current_user_input_text")
    ocr_result_text = state.get("ocr_extracted_text")
    ocr_error = state.get("ocr_error_message")

    if not llm_openai_text_responder:
        logger.error("[AGENT TEXT] OpenAI Text LLM (llm_openai_text_responder) not initialized.")
        new_state["final_llm_response"] = "Error: The AI text responder service is unavailable."
        return new_state

    llm_input_parts = []
    if user_input_text and user_input_text.strip():
        llm_input_parts.append(f"User's typed message: \"{user_input_text.strip()}\"")

    if ocr_error:
        llm_input_parts.append(f"Note regarding the uploaded document: {ocr_error}")
    elif ocr_result_text and ocr_result_text.strip():
        llm_input_parts.append(f"The following text was extracted from the uploaded document:\n--- DOCUMENT START ---\n{ocr_result_text.strip()}\n--- DOCUMENT END ---")

    if not llm_input_parts:
        logger.warning("[AGENT TEXT] No user text or OCR data to process.")
        new_state["final_llm_response"] = "I'm ready to help! Please type your medical question or upload a relevant document."
        return new_state

    final_input_for_llm = "\n\n".join(llm_input_parts)
    logger.debug(f"[AGENT TEXT] Combined input for LLM (first 300 chars): {final_input_for_llm[:300]}...")

    system_message = """
You are a specialized AI medical assistant named "MediCare Guide". Your primary purpose is to provide guidance, information, and support exclusively related to health, medical conditions, symptoms, treatments, general healthcare topics, and medical terminology.
If text from an uploaded document is provided, acknowledge the document and base your response on its content, in conjunction with any typed user message.
If an error regarding document processing is noted, politely inform the user about the issue with their document and that you cannot use its content.

**Your core responsibilities are:**
1.  **Focus on Health and Medical Topics:** Respond only to questions and statements directly pertaining to health.
2.  **Encourage Medical Dialogue:** Gently guide users towards discussing their specific health concerns.
3.  **Deflect Unrelated Topics Politely:** If a user asks about non-medical subjects, decline and redirect. Example: "My apologies, but my expertise is limited to medical and healthcare topics. How can I assist you with your health concerns?"
4.  **Maintain a Professional and Empathetic Tone:** Be respectful, clear, and empathetic.
5.  **No Diagnoses or Prescriptions:** ALWAYS state that you are an AI assistant and users MUST consult a qualified healthcare professional for personal medical advice, diagnosis, or treatment. Example: "Please remember, I am an AI assistant and cannot provide medical diagnoses or treatment. Always consult with a doctor."

**Negative Constraints (What NOT to do):**
* Do NOT engage in conversations about non-medical topics.
* Do NOT generate creative content like stories or poems.
* Do NOT discuss politics or personal opinions.

If someone asks who is your builder, respond with: "I am an AI assistant developed by the StackToServ team, dedicated to building intelligent AI systems."
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}")
    ])

    try:
        logger.debug("[AGENT TEXT] Invoking OpenAI Text LLM (llm_openai_text_responder)...")
        response_obj = llm_openai_text_responder.invoke(prompt.format_messages(input=final_input_for_llm))
        response_content = response_obj.content
        logger.debug(f"[AGENT TEXT] LLM generated response (first 100 chars): '{response_content[:100]}...'")
        new_state["final_llm_response"] = response_content
    except Exception as e:
        logger.exception(f"[AGENT TEXT] ERROR during OpenAI LLM invocation: {e}")
        new_state["final_llm_response"] = f"An error occurred while trying to generate a response: {str(e)[:100]}..."

    logger.debug("[AGENT TEXT] === EXITING text_responder_agent ===")
    return new_state


# --- Define the LangGraph Workflow ---
workflow = StateGraph(AgentState)

workflow.add_node("ocr_processor_node_openai", ocr_agent_openai) # Renamed for clarity
workflow.add_node("text_responder_node", text_responder_agent)

workflow.set_entry_point("ocr_processor_node_openai")
workflow.add_edge("ocr_processor_node_openai", "text_responder_node")
workflow.add_edge("text_responder_node", END)

try:
    compiled_medical_assistant_graph = workflow.compile()
    logger.info("LangGraph with OpenAI OCR and Text agents compiled successfully.")
except Exception as e:
    logger.exception("Failed to compile LangGraph:")
    compiled_medical_assistant_graph = None
# AI Medical Assistant (MediCare Guide)

## 📖 Description

MediCare Guide is an AI-powered medical assistant built with Django and LangGraph. It helps users understand health conditions, interpret medical documents (lab reports via OCR), and get information using a multi-agent system that features Retrieval Augmented Generation (RAG), multilingual support (English/Persian), and doctor recommendations.

---

## ✨ Features

* Web-based conversational UI.
* **Multi-Agent System (LangGraph):**
    * **Triage & Greeting:** Handles initial interaction, routes queries.
    * **OCR:** Extracts structured text from PDFs/images (e.g., lab results).
    * **Reasoning:** Deep analysis with RAG from a medical knowledge base.
    * **Revisor & Output:** Refines analysis, manages persona, handles language, integrates doctor suggestions.
* **Retrieval Augmented Generation (RAG):** Uses a FAISS vector store (built with `sentence-transformers/all-mpnet-base-v2`) for knowledge grounding.
* **Doctor Recommendation:** Suggests doctors from a local JSON file based on analyzed health issues.
* **Multilingual:** Supports English and Persian.
* **Document Upload:** Accepts PDF and image files.
* **Conversational Memory:** Maintains session context.

---

## 🛠️ Tech Stack

* **Backend:** Django, SQLite
* **Agent Orchestration:** LangGraph
* **LLMs:** OpenAI GPT-4o (Triage, OCR, Revisor), GPT-o3 (Reasoning)
* **Embeddings (RAG):** Hugging Face `sentence-transformers/all-mpnet-base-v2`
* **Vector Store (RAG):** FAISS
* **Key Libraries:** `langchain`, `django`, `python-dotenv`, `sentence-transformers`, `faiss-cpu`, `pdf2image`

---

## ⚙️ Prerequisites

* Python 3.9+ & Pip
* Poppler (for PDF OCR via `pdf2image`)
* An OpenAI API Key

---

## 🚀 Setup and Installation

1.  **Clone Repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[Your GitHub Username]/[Your Repo Name].git
    cd [Your Repo Name]
    ```

2.  **Create & Activate Virtual Environment:**
    ```bash
    python -m venv .venv
    # Windows: .\.venv\Scripts\activate
    # macOS/Linux: source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    * Ensure you have a `requirements.txt` (create one with `pip freeze > requirements.txt` after installing packages listed in the Tech Stack section, e.g., `pip install django langchain langchain-openai ...`).
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Variables (`.env` file in project root):**
    ```env
    OPENAI_API_KEY="sk-your_openai_api_key_here"
    ```

5.  **Data Files:**
    * Create a `rag_data/` folder in the project root. Place your `index.faiss` and `index.pkl` (FAISS index built with `sentence-transformers/all-mpnet-base-v2`) inside it.
    * Place your `doctors_list.json` file in the project root.

6.  **Django Setup:**
    ```bash
    python manage.py makemigrations medassist
    python manage.py migrate
    python manage.py createsuperuser
    ```

---

## ▶️ Running the Application

1.  Start the Django development server:
    ```bash
    python manage.py runserver
    ```
2.  Open your browser to `http://127.0.0.1:8000/medassist/chat/` (or your configured chat URL).
3.  Log in with your superuser credentials if prompted.

---

## 💬 Usage

Interact with "MediCare Guide" by typing messages or uploading medical documents (PDFs, images). The assistant will process your input through its agent pipeline:

1.  **Triage:** Initial greeting and determines if the query is general or medical.
2.  **OCR (if file uploaded & medical):** Extracts text from documents.
3.  **Reasoning (if medical):** Analyzes query, OCR data, chat history, and retrieves relevant information from the medical knowledge base (RAG).
4.  **Revisor (if medical):** Refines the analysis, suggests doctors if a health issue is identified and relevant doctors are found, handles language (English/Persian), and crafts the final response with disclaimers.

---

## 🔧 Configuration Highlights

* **API Keys:** In `.env` file.
* **RAG Index Path:** `RAG_INDEX_FOLDER_PATH` in `medassist/agents/medical_assistant_graph.py` (defaults to `PROJECT_ROOT/rag_data/`).
* **Doctor Data:** `DOCTORS_JSON_PATH` in `medassist/agents/medical_assistant_graph.py` (defaults to `PROJECT_ROOT/doctors_list.json`).
* **LLM Models:** Defined in `medical_assistant_graph.py`.

---

## 🤝 Contributing (Optional Placeholder)

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.

---

## 📜 License (Optional Placeholder)

This project is licensed under the MIT.

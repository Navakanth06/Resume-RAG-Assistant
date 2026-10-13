# 📄 Resume RAG Assistant

An AI-powered Resume Question Answering application built using **Retrieval-Augmented Generation (RAG)**. Users can upload a resume in PDF format and ask questions about its content. The application retrieves the most relevant information from the uploaded resume and generates accurate, context-aware responses using **Google Gemini**.

---

## 🚀 Features

- Upload Resume (PDF)
- Extract text from PDF
- Intelligent text chunking
- Generate semantic embeddings
- Store embeddings in ChromaDB
- Retrieve relevant resume sections
- Context-aware question answering using Gemini
- Simple and interactive Streamlit interface

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Google Gemini API
- Sentence Transformers
- ChromaDB
- LangChain Text Splitter
- PyPDF

---

## 📂 Project Structure

```text
Resume-RAG-Assistant/

├── streamlit_app.py
├── pdf_processor.py
├── chunking.py
├── embeddings.py
├── vector_store.py
├── rag.py
├── requirements.txt
├── README.md
└── .env
```

---

## ▶️ Getting Started

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project folder with your Gemini API key:

   ```text
   GEMINI_API_KEY=your_api_key_here
   ```

   When deploying on Streamlit Cloud, add `GEMINI_API_KEY` under the app's **Secrets** instead.

3. Run the app:

   ```bash
   streamlit run streamlit_app.py
   ```

Uploaded resumes are kept in memory only for the current session. Nothing is written to disk.

---

## 💡 Example Questions

- Summarize my resume.
- What technical skills do I have?
- What projects have I completed?
- What certifications are listed?
- What programming languages do I know?
- What is my educational background?
- Suggest interview questions based on my resume.

---

## 🎯 Future Improvements

- Conversation memory
- Multi-document support
- Chat history
- Source citation
- Resume comparison
- Voice input
- Deployment on Streamlit Cloud

---

## 👨‍💻 Author

**Navakanth**

If you found this project useful, consider giving it a ⭐ on GitHub.

# 📘 PDF Research Assistant (RAG)

A **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents and ask questions that are answered **strictly from the uploaded content**. The system is fully **local**, powered by **Ollama (LLaMA3)**, **FAISS**, and **Streamlit**.

This project is ideal for:

* 📄 Research paper analysis
* 📚 Book & notes Q&A
* 🤖 Generative AI portfolio projects
* 🧠 Interview demonstrations (RAG, LLMs, embeddings)

---

## 🚀 Features

* Upload **any PDF** (book, notes, research paper)
* Automatic text chunking
* Semantic search using **FAISS**
* Embeddings via **nomic-embed-text**
* Answers generated using **LLaMA3**
* Prevents hallucinations by answering **only from retrieved context**
* Simple and clean **Streamlit UI**
* Fully **offline / local** (no OpenAI API)

---

## 🏗️ Project Architecture

PDF → Chunking → Embeddings → FAISS Index
                     ↓
                User Question
                     ↓
              Relevant Chunks
                     ↓
                 LLaMA Answer

## 📁 Project Structure

AI_RESEARCH_ASSISTANT/
│
├── app.py            # Streamlit UI
├── rag_core.py       # Core RAG logic
├── requirements.txt  # Dependencies
└── README.md


## 🧰 Tech Stack

| Component   | Technology          |
| ----------- | ------------------- |
| UI          | Streamlit           |
| LLM         | LLaMA3 (via Ollama) |
| Embeddings  | nomic-embed-text    |
| Vector DB   | FAISS               |
| PDF Parsing | PyPDF               |
| Language    | Python              |

---

## ⚙️ Setup Instructions

### 1️⃣ Install Ollama

Download and install Ollama from:

> [https://ollama.com](https://ollama.com)

Pull required models:

ollama pull llama3
ollama pull nomic-embed-text
### 2️⃣ Create Conda Environment (Recommended)

conda create -n llama_rag python=3.10 -y
conda activate llama_rag

### 3️⃣ Install Dependencies

pip install -r requirements.txt

### 4️⃣ Run the Application

streamlit run app.py


The app will open in your browser at:

http://localhost:8501

## 🧪 How to Use

1. Upload a **PDF document**
2. Wait for indexing to complete
3. Ask questions related to the document
4. View answers with cited chunk sources

---

## 🛡️ Hallucination Control

The assistant is explicitly instructed to:

* Answer **only from retrieved context**
* Respond with *"I don't know"* if the answer is not present in the PDF

This ensures factual and trustworthy outputs.

## 🔮 Future Enhancements

* Multi-PDF support
* Persistent FAISS index
* Chat history & memory
* Highlight answers in PDF
* Dockerization
* Authentication





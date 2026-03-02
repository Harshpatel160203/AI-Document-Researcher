# 📄 AI Document Researcher (RAG-based)

A production-ready **Retrieval-Augmented Generation (RAG)** application that allows users to "chat" with their PDF documents. Built with **LangChain**, **Groq (Llama 3.3)**, and **Streamlit**.

## 🚀 Live Demo
[Insert your Streamlit URL here, e.g., https://harsh-ai-researcher.streamlit.app]

## 🛠️ The Architecture (RAG)
This project implements a full RAG pipeline to overcome LLM context windows and provide accurate, document-based answers:
1. **Ingestion**: PDF data is loaded and split into semantic chunks using `RecursiveCharacterTextSplitter`.
2. **Embeddings**: Chunks are converted into vector representations using `HuggingFaceEmbeddings` (all-MiniLM-L6-v2).
3. **Vector Store**: High-dimensional vectors are stored in a **ChromaDB** local instance for fast similarity search.
4. **Retrieval**: The system fetches the most relevant context based on user queries.
5. **Generation**: A specialized prompt passes the context to **Llama 3.3 (via Groq)** for a concise, grounded response.



## 💻 Tech Stack
- **Frontend**: Streamlit
- **Orchestration**: LangChain (LCEL)
- **LLM**: Groq (Llama-3.3-70b-versatile)
- **Embeddings**: HuggingFace Transformers
- **Database**: ChromaDB

## ⚙️ Local Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/Harshpatel160203/AI-Document-Researcher.git](https://github.com/Harshpatel160203/AI-Document-Researcher.git)
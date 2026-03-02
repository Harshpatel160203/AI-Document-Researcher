import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# 1. Load Keys
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# 2. UI Setup
st.set_page_config(page_title="Zenith AI Researcher", layout="wide")
st.title("📄 AI Document Researcher")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# EVERYTHING below this line must be indented to be inside the "if"
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # 3. RAG Pipeline
    loader = PyPDFLoader("temp.pdf")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    
    # FIX: Use a proper embedding model here
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # 4. The LLM Setup (Indented inside the IF)
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")

    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(
    """
    You are a professional Research Assistant. 
    Use the following pieces of retrieved context to answer the question.
    If the answer is not in the context, say "I cannot find the answer in the uploaded document." 
    Do not make up facts.
    
    Context: {context}
    Question: {input}
    Answer:"""
)

    # 5. The LCEL Chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    user_question = st.text_input("Ask a question about your PDF:")
    
    if user_question:
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(user_question)
            st.write("### Answer:")
            st.write(response)
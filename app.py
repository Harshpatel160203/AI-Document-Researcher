import os
import ssl

# SSL Bypass for local Windows machines
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'
ssl._create_default_https_context = ssl._create_unverified_context

import streamlit as st
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
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

if uploaded_file:
    # Save file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # 3. RAG Pipeline Status
    with st.status("🚀 Processing document...", expanded=True) as status:
        st.write("🔍 Loading PDF...")
       # Switch from PyPDFLoader to Unstructured
        loader = UnstructuredPDFLoader(tmp_file_path, strategy="fast")
        data = loader.load()
        
        st.write("✂️ Creating semantic chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(data)
        
        st.write("🧠 Generating Vector Embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        st.write("📁 Initializing Vector Database...")
        vectorstore = Chroma.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        status.update(label="✅ Document Ready!", state="complete", expanded=False)

    # 4. LLM & Prompt Setup
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")

    prompt = ChatPromptTemplate.from_template(
    """
    You are an expert Document Analyst. 
    The following context is from the resume of Harsh Balkrishna Patel. 
    If a question asks about his education, skills, or experience, use the information below.

    Context: {context}
    Question: {input}
    
    Format your response exactly like this:
    The final answer is: [answer]
    """
)

# 5. The LCEL Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. Chat Interface
    user_question = st.text_input("Ask a question about your PDF:")
    
    if user_question:
        with st.spinner("Analyzing..."):
            response = rag_chain.invoke(user_question)
            
            # This ensures the output matches your required "Boxed" style
            st.write("### Answer:")
            st.code(response) # Using .code() makes it look like a final result

    # Cleanup temp file
    os.remove(tmp_file_path)
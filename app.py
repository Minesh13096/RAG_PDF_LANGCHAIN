
"""
PDF RAG Chatbot — LangChain + FAISS + HuggingFace
-------------------------------------------------------------
End-to-end Retrieval-Augmented Generation in one file.
 
RAG steps in this app:
  1. LOAD    - read the uploaded PDF into text (PyPDFLoader)
  2. CHUNK   - split text into small overlapping pieces (RecursiveCharacterTextSplitter)
  3. EMBED   - turn each chunk into a vector (sentence-transformers/all-MiniLM-L6-v2)
  4. STORE   - save the vectors in a FAISS vector database
  5. RETRIEVE- find the chunks most similar to the question
  6. AUGMENT - put those chunks into the prompt as context
  7. GENERATE- ask the HuggingFace LLM to answer using that context
"""
 
import os
import tempfile
import streamlit as st
 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    ChatHuggingFace,
    HuggingFaceEndpoint,
)
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
 
# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")
st.title("📄 PDF Q&A — RAG Chatbot")
st.caption("LangChain + FAISS + HuggingFace")
 
# ----------------------------------------------------------------------
# Read the HuggingFace token.
# On Streamlit Cloud it comes from "Secrets"; locally from your environment
# or from .streamlit/secrets.toml
# ----------------------------------------------------------------------
def get_hf_token():
    # Try Streamlit secrets first, but don't crash if no secrets.toml exists.
    try:
        if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
            return st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    except Exception:
        pass
    # Fall back to an environment variable (works locally without a secrets file)
    return os.getenv("HUGGINGFACEHUB_API_TOKEN")
 
HF_TOKEN = get_hf_token()
if HF_TOKEN:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
else:
    st.error(
        "No HuggingFace token found. Either create .streamlit/secrets.toml with "
        "HUGGINGFACEHUB_API_TOKEN, or set it as an environment variable."
    )
    st.stop()
 
# ----------------------------------------------------------------------
# Models — cached with @st.cache_resource so they load only once
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    # Light, fast, free embedding model that runs locally
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
 
@st.cache_resource(show_spinner=False)
def get_llm():
    # Free-tier HuggingFace LLM called through the Inference API
    endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",   # light instruct model
        task="text-generation",
        max_new_tokens=512,
        temperature=0.2,
        provider="auto",                       # let HF pick an available provider
    )
    return ChatHuggingFace(llm=endpoint)
 
# ----------------------------------------------------------------------
# Build the FAISS vector store from the uploaded PDF.
# Cached per file so re-asking questions doesn't re-index every time.
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Reading & indexing your PDF...")
def build_vectorstore(pdf_bytes: bytes):
    # Save uploaded bytes to a temp file so PyPDFLoader can read from a path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
 
    # STEP 1: LOAD
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
 
    # STEP 2: CHUNK
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(pages)
 
    # STEP 3 + 4: EMBED each chunk and STORE in FAISS
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
 
    return vectorstore, len(chunks)
 
# ----------------------------------------------------------------------
# UI: upload a PDF and ask questions
# ----------------------------------------------------------------------
uploaded = st.file_uploader("Upload a PDF to use as the knowledge base", type="pdf")
 
if uploaded:
    vectorstore, n_chunks = build_vectorstore(uploaded.getvalue())
    st.success(f"Indexed {n_chunks} chunks. Ask me anything about this PDF!")
 
    question = st.text_input("Your question:")
 
    if question:
        with st.spinner("Thinking..."):
            # STEP 5: RETRIEVE the top 3 most relevant chunks
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(question)
            context = "\n\n".join(d.page_content for d in docs)
 
            # STEP 6: AUGMENT — build the prompt with the retrieved context
            llm = get_llm()
            messages = [
                SystemMessage(content=(
                    "You are a helpful assistant. Answer the question using ONLY the "
                    "context below. If the answer is not in the context, say you don't know.\n\n"
                    f"Context:\n{context}"
                )),
                HumanMessage(content=question),
            ]
 
            # STEP 7: GENERATE the final answer
            answer = llm.invoke(messages).content
 
        st.markdown("### Answer")
        st.write(answer)
 
        # Show the source chunks so you can verify the answer
        with st.expander("See the chunks used as context"):
            for i, d in enumerate(docs, 1):
                page = d.metadata.get("page", "?")
                st.markdown(f"**Chunk {i}** (page {page})")
                st.write(d.page_content)
else:
    st.info("👆 Upload a PDF to get started.")
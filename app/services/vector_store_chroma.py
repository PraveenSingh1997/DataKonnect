# app/services/vector_store.py

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Constants
VECTOR_DIR = "./chroma_store"
DOC_PATH = r"C:\Users\praveen.singh1\Desktop\Project_Src_code\documents\lmg_document.txt"

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_vectorstore_retriever():
    """
    Returns a Chroma retriever object (builds vector store if it doesn't exist).
    """
    if not os.path.exists(f"{VECTOR_DIR}/index"):
        print("🔄 Creating vector store from documents...")
        loader = TextLoader(DOC_PATH, encoding='utf-8')
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(chunks, embedding_model, persist_directory=VECTOR_DIR)
        vectorstore.persist()
    else:
        
        print("✅ Loading existing vector store...")
        vectorstore = Chroma(persist_directory=VECTOR_DIR, embedding_function=embedding_model)

    return vectorstore.as_retriever(search_kwargs={"k": 3})


get_vectorstore_retriever()
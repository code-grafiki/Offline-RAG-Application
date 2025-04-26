import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHROMA_DIR = "chroma_db"

embedding_model = OllamaEmbeddings(model="nomic-embed-text")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_vectorstore():
    ensure_dir(CHROMA_DIR)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
        collection_name="documents"
    )

def get_retriever():
    return get_vectorstore().as_retriever()

def ingest_documents(docs, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)
    if not chunks:
        return 0
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    vectorstore.persist()
    return len(chunks)

def clear_knowledgebase():
    try:
        chroma = get_vectorstore()
        chroma._collection = None
        del chroma
    except Exception:
        pass
    if os.path.exists(CHROMA_DIR):
        try:
            shutil.rmtree(CHROMA_DIR)
        except Exception:
            pass
    return True
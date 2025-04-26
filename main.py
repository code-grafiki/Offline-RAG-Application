import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from db_utils import get_retriever, ingest_documents, clear_knowledgebase, get_vectorstore
from file_loader import load_file
from langchain.docstore.document import Document
import json
import os
import requests
import shutil
from collections import defaultdict
from datetime import datetime

# -------- CONFIG --------
CONFIG_FILE = "user_config.json"
DEFAULT_CONFIG = {
    "model_name": "llama3",
    "temperature": 0.7,
    "top_p": 1.0,
    "max_tokens": 512,
    "persona": "You are a helpful assistant.",
    "chunk_size": 500,
    "chunk_overlap": 100
}

# --------- HELPERS ---------
@st.cache_resource
def get_ollama_llm(model_name, temperature, top_p, max_tokens):
    return Ollama(
        model=model_name,
        temperature=temperature,
        top_p=top_p
    )

@st.cache_resource(ttl=None)
def get_cached_retriever():
    return get_retriever()

def clear_cached_resources():
    # Clear all st.cache_resource entries
    st.cache_resource.clear()

def get_knowledge_base_stats():
    """Get statistics about the knowledge base."""
    try:
        vectorstore = get_vectorstore()
        collection = vectorstore._collection
        if collection:
            collection_data = collection.get()
            # Count unique documents by their source metadata
            metadatas = collection_data.get("metadatas", [])
            unique_sources = len({meta.get("source", "") for meta in metadatas if meta})
            return {
                "document_count": unique_sources,
                "embeddings_count": len(collection_data.get("ids", [])),
                "last_updated": datetime.now().isoformat()
            }
    except Exception as e:
        print(f"Error getting stats: {e}")  # Debug print
        return {"document_count": 0, "embeddings_count": 0, "last_updated": None}

def create_rag_chain(llm, retriever):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

def enhance_answer_with_context(answer, sources, llm, prompt_prefix):
    """Enhance the answer with additional context and explanations using the LLM."""
    if not sources:
        return answer

    # Format the sources for context
    source_texts = []
    for doc in sources:
        source = doc.metadata.get("source", "Unknown")
        content = doc.page_content
        source_texts.append(f"From {source}:\n{content}")

    # Create a prompt for the LLM to enhance the answer
    enhancement_prompt = f"""
    {prompt_prefix}

    Original Answer: {answer}

    Context from Knowledge Base:
    {chr(10).join(source_texts)}

    Please enhance the original answer by:
    1. Adding more context and details from the provided sources
    2. Explaining any technical terms or concepts
    3. Providing examples or analogies where relevant
    4. Making the answer more comprehensive and informative
    5. Maintaining accuracy and staying within the scope of the provided context

    Enhanced Answer:
    """

    try:
        enhanced_answer = llm.invoke(enhancement_prompt)
        return enhanced_answer
    except Exception as e:
        print(f"Error enhancing answer: {e}")
        return answer

def load_user_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return DEFAULT_CONFIG

def save_user_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)

def get_installed_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        st.sidebar.warning(f"Could not fetch models from Ollama: {e}")
    return ["llama3", "mistral", "gemma"]  # fallback list

# --------- UI ---------
st.set_page_config(page_title="🧠 Local RAG App", layout="wide")
st.title("🧠 Offline RAG Application")

config = load_user_config()

# Sidebar - Model settings
st.sidebar.title("🔧 Model Settings")
model_list = get_installed_ollama_models()
default_model = config.get("model_name", model_list[0])
if default_model not in model_list:
    default_model = model_list[0]
model_name = st.sidebar.radio("Choose Model", model_list, index=model_list.index(default_model))
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, config.get("temperature", 0.7), step=0.05)
top_p = st.sidebar.slider("Top P", 0.0, 1.0, config.get("top_p", 1.0), step=0.05)
max_tokens = st.sidebar.slider("Max Tokens", 64, 2048, config.get("max_tokens", 512), step=64)
persona_options = {
    "Helpful Assistant": "You are a helpful assistant.",
    "Technical Expert": "You are a technical expert that explains with precision.",
    "Friendly Guide": "You are a friendly and approachable tutor.",
    "Summarizer": "You summarize content concisely and clearly."
}
persona_label = st.sidebar.selectbox("System Persona", list(persona_options.keys()), index=0)
prompt_prefix = persona_options[persona_label]

chunk_size = st.sidebar.slider("Chunk Size", 100, 2000, config.get("chunk_size", 500), step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, config.get("chunk_overlap", 100), step=10)

# Save user config for reuse
save_user_config({
    "model_name": model_name,
    "temperature": temperature,
    "top_p": top_p,
    "max_tokens": max_tokens,
    "persona": prompt_prefix,
    "chunk_size": chunk_size,
    "chunk_overlap": chunk_overlap
})

# Load LLM and Retriever
llm = get_ollama_llm(model_name, temperature, top_p, max_tokens)
retriever = get_cached_retriever()
rag_chain = create_rag_chain(llm, retriever)

# File Upload and Ingestion
st.sidebar.markdown("---")
st.sidebar.header("📄 Knowledge Base")

# Knowledge Base Stats
kb_stats = get_knowledge_base_stats()
st.sidebar.markdown(f"""
### 📊 Knowledge Base Stats
- Documents: {kb_stats['document_count']}
- Embeddings: {kb_stats['embeddings_count']}
- Last Updated: {kb_stats['last_updated'] or 'Never'}
""")

# Initialize session state variables
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "last_processed_file" not in st.session_state:
    st.session_state.last_processed_file = None

# File Upload
uploaded_file = st.sidebar.file_uploader(
    "Upload a file", 
    type=["pdf", "txt", "md", "docx", "xlsx", "pptx", "html", "png", "jpg", "jpeg"]
)

# Process new file if uploaded
if uploaded_file is not None and uploaded_file.name != st.session_state.last_processed_file:
    try:
        # Create temp directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        temp_path = os.path.join("temp", uploaded_file.name)
        
        # Save the uploaded file
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process the file
        with st.spinner("Processing document..."):
            docs = load_file(temp_path)
            chunks = ingest_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            # Update session state
            st.session_state.uploaded_files.append(uploaded_file.name)
            st.session_state.last_processed_file = uploaded_file.name
            
            # Show success message
            st.sidebar.success(f"Ingested {chunks} chunks from {uploaded_file.name}")
            
            # Force a rerun to update the UI
            st.rerun()
            
    except Exception as e:
        st.sidebar.error(f"Error processing file: {e}")
        # Reset the last processed file on error
        st.session_state.last_processed_file = None

# Show uploaded files
if st.session_state.uploaded_files:
    st.sidebar.markdown("#### 📚 Current Documents")
    for idx, file_name in enumerate(st.session_state.uploaded_files):
        col1, col2 = st.sidebar.columns([3, 1])
        col1.markdown(f"`{file_name}`")
        if col2.button("🗑️", key=f"remove_{idx}_{file_name}"):
            try:
                # Remove document from vectorstore
                vectorstore = get_vectorstore()
                # Get all documents with this source
                docs = vectorstore.get(where={"source": file_name})
                if docs and docs.get("ids"):
                    # Delete by IDs
                    vectorstore.delete(ids=docs["ids"])
                st.session_state.uploaded_files.remove(file_name)
                if st.session_state.last_processed_file == file_name:
                    st.session_state.last_processed_file = None
                st.sidebar.success(f"Removed {file_name} from knowledge base")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Failed to remove {file_name}: {e}")

# Clear and Refresh Buttons
col1, col2 = st.sidebar.columns(2)
if col1.button("🧩 Clear Knowledge Base"):
    try:
        clear_cached_resources()
        clear_knowledgebase()
        st.session_state.uploaded_files = []
        st.session_state.last_processed_file = None
        st.sidebar.success("Knowledge base contents cleared!")
        st.sidebar.toast("🌧️ Knowledge base reset!", icon="🚮")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Failed to clear knowledge base: {e}")
        if "Permission denied" in str(e):
            st.sidebar.warning("💡 If the error persists, please:\n1. Stop the Streamlit app\n2. Delete the chroma_db folder manually\n3. Restart the app")

if col2.button("🔄 Refresh Knowledge Base"):
    if st.session_state.last_processed_file:
        try:
            clear_cached_resources()
            clear_knowledgebase()
            temp_path = os.path.join("temp", st.session_state.last_processed_file)
            docs = load_file(temp_path)
            chunks = ingest_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            st.sidebar.success(f"Knowledge base refreshed with {chunks} chunks!")
            st.sidebar.toast("♻️ Refreshed using last uploaded file", icon="🔁")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Failed to refresh: {e}")
    else:
        st.sidebar.warning("⚠️ No previously uploaded file to refresh with")

# Chat History State
if "history" not in st.session_state:
    st.session_state.history = []

# Toggle between LLM only vs RAG
use_kb = st.radio("Answer mode", ["With Knowledge Base", "LLM Only"], horizontal=True) == "With Knowledge Base"

# Main UI
user_query = st.text_input("🔍 Ask a question about your documents:")
if st.button("Submit") and user_query:
    with st.spinner("Generating answer..."):
        from_kb = False  # Initialize from_kb variable
        try:
            if use_kb:
                result = rag_chain.invoke({"query": user_query})
                sources = result.get("source_documents", [])
                relevant_sources = [doc for doc in sources if doc.page_content.strip()]
                from_kb = len(relevant_sources) > 0 and "i don't know" not in result["result"].lower()
                
                if from_kb:
                    base_answer = result["result"]
                    # Enhance the answer with additional context
                    enhanced_answer = enhance_answer_with_context(
                        base_answer, 
                        relevant_sources, 
                        llm, 
                        prompt_prefix
                    )
                    provenance = "📚 Retrieved from Knowledge Base and Enhanced with LLM"
                    
                    # Format the answer with sources
                    st.markdown("### 📿 Answer")
                    st.success(enhanced_answer)
                    st.caption(provenance)
                    
                    if relevant_sources:
                        st.markdown("### 📚 Sources")
                        grouped = defaultdict(list)
                        for doc in relevant_sources:
                            source = doc.metadata.get("source", "Unknown")
                            grouped[source].append(doc.page_content)
                        
                        for i, (source, chunks) in enumerate(grouped.items(), 1):
                            with st.expander(f"Source {i}: {source}"):
                                for j, chunk in enumerate(chunks, 1):
                                    st.markdown(f"**Chunk {j}:**")
                                    st.markdown(f"> {chunk}")
                else:
                    answer = llm.invoke(f"{prompt_prefix}\n\n{user_query}")
                    relevant_sources = []
                    provenance = "🧠 Generated from LLM only"
                    st.markdown("### 📿 Answer")
                    st.success(answer)
                    st.caption(provenance)
            else:
                answer = llm.invoke(f"{prompt_prefix}\n\n{user_query}")
                relevant_sources = []
                provenance = "🧠 Generated from LLM only"
                st.markdown("### 📿 Answer")
                st.success(answer)
                st.caption(provenance)

            st.session_state.history.append((user_query, enhanced_answer if from_kb else answer, relevant_sources, provenance))

        except Exception as e:
            st.error(f"Error: {e}")

# Show chat history
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 💬 Chat History")
    for i, (q, a, srcs, prov) in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"> {a}")
        st.caption(prov)
        if srcs:
            with st.expander("Sources"):
                grouped = defaultdict(list)
                for doc in srcs:
                    src = doc.metadata.get("source", "Unknown")
                    grouped[src].append(doc.page_content)
                for j, (src, chunks) in enumerate(grouped.items(), 1):
                    st.markdown(f"**{j}.** `{src}` ({len(chunks)} chunks)")
                    st.markdown(f"> {chunks[0][:300]}...")

st.markdown("---")
st.markdown("*Model: {} | Temp: {} | Top P: {} | Max Tokens: {}*".format(model_name, temperature, top_p, max_tokens))

# Local RAG Application

A local RAG (Retrieval-Augmented Generation) application built with Streamlit and LangChain. This application allows you to create a local knowledge base from your documents and interact with it using various LLM models through Ollama.

## Features

- Document ingestion support for multiple file formats (PDF, TXT, DOCX, etc.)
- Local LLM integration using Ollama
- Semantic search and retrieval from your knowledge base
- Configurable model parameters (temperature, top_p, max_tokens)
- Customizable system personas for different response styles
- Knowledge base statistics and management
- Real-time document processing and indexing

## Prerequisites

- Python 3.8 or higher
- Ollama installed and running locally
- Git (for cloning the repository)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/local_rag_app.git
cd local_rag_app
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Install and set up Ollama:
```bash
# Download and install Ollama from https://ollama.com/
# Pull a model (e.g., llama3)
ollama pull llama3
```

## Running the Application

1. Start Ollama (if not already running):
```bash
ollama serve
```

2. In a new terminal, run the Streamlit app:
```bash
streamlit run main.py
```

3. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Usage

1. Upload documents through the sidebar interface
2. Configure model settings as needed
3. Ask questions about your documents
4. View and manage your knowledge base

## License

MIT License 
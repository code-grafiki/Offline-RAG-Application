from langchain.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader, UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader, UnstructuredMarkdownLoader
)
from langchain.docstore.document import Document
import pytesseract
from PIL import Image
import os
import re
from datetime import datetime

def clean_text(text: str) -> str:
    """Clean and preprocess text content."""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\'"]', '', text)
    return text.strip()

def extract_metadata(file_path: str) -> dict:
    """Extract metadata from file."""
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    last_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
    
    return {
        "source": file_name,
        "file_size": file_size,
        "last_modified": last_modified.isoformat(),
        "file_type": os.path.splitext(file_path)[1].lower()
    }

def load_pdf(file_path: str):
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        metadata = extract_metadata(file_path)
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata.update(metadata)
        return docs
    except Exception as e:
        raise ValueError(f"Failed to load PDF: {e}")

def load_text(file_path: str):
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load()
        metadata = extract_metadata(file_path)
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata.update(metadata)
        return docs
    except Exception as e:
        raise ValueError(f"Failed to load text file: {e}")

def load_word(file_path: str):
    try:
        loader = UnstructuredWordDocumentLoader(file_path)
        docs = loader.load()
        metadata = extract_metadata(file_path)
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata.update(metadata)
        return docs
    except Exception as e:
        raise ValueError(f"Failed to load Word document: {e}")

def load_excel(file_path: str):
    try:
        loader = UnstructuredExcelLoader(file_path)
        docs = loader.load()
        metadata = extract_metadata(file_path)
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata.update(metadata)
        return docs
    except Exception as e:
        raise ValueError(f"Failed to load Excel file: {e}")

def load_powerpoint(file_path: str):
    try:
        loader = UnstructuredPowerPointLoader(file_path)
        docs = loader.load()
        metadata = extract_metadata(file_path)
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata.update(metadata)
        return docs
    except Exception as e:
        raise ValueError(f"Failed to load PowerPoint file: {e}")

def load_html(file_path: str):
    try:
        loader = UnstructuredHTMLLoader(file_path)
        docs = loader.load()
        metadata = extract_metadata(file_path)
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata.update(metadata)
        return docs
    except Exception as e:
        raise ValueError(f"Failed to load HTML file: {e}")

def load_markdown(file_path: str):
    try:
        loader = UnstructuredMarkdownLoader(file_path)
        docs = loader.load()
        metadata = extract_metadata(file_path)
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata.update(metadata)
        return docs
    except Exception as e:
        raise ValueError(f"Failed to load Markdown file: {e}")

def load_image(file_path: str):
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        metadata = extract_metadata(file_path)
        doc = Document(page_content=clean_text(text), metadata=metadata)
        return [doc]
    except Exception as e:
        raise ValueError(f"Failed to process image: {e}")

def load_file(file_path: str):
    """Load and process a file based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    loaders = {
        ".pdf": load_pdf,
        ".txt": load_text,
        ".docx": load_word,
        ".xlsx": load_excel,
        ".pptx": load_powerpoint,
        ".html": load_html,
        ".md": load_markdown,
        ".png": load_image,
        ".jpg": load_image,
        ".jpeg": load_image
    }
    
    if ext in loaders:
        return loaders[ext](file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
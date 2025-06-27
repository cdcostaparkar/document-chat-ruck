from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
from typing import List
import os
import tempfile
from pathlib import Path
import unicodedata
import string

# LangChain imports
from langchain_community.document_loaders import Docx2txtLoader, CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

app = FastAPI(title="Document RAG API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # NextJS default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

class IngestResponse(BaseModel):
    message: str
    files_processed: int

# Global variables for models
embedding = None
vectorstore = None
llm = None
persist_dir = "chroma_db"

def initialize_models():
    """Initialize the embedding model and LLM"""
    global embedding, llm
    try:
        embedding = OllamaEmbeddings(model="nomic-embed-text")
        llm = OllamaLLM(model="qwen3:0.6b")
        print("Models initialized successfully")
    except Exception as e:
        print(f"Error initializing models: {e}")
        raise

def load_vectorstore():
    """Load existing vectorstore if it exists"""
    global vectorstore, embedding
    if os.path.exists(persist_dir) and embedding:
        try:
            vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=embedding
            )
            print("Vectorstore loaded successfully")
        except Exception as e:
            print(f"Error loading vectorstore: {e}")
            vectorstore = None

class TextPreprocessor:
    """Advanced text preprocessing for better RAG performance"""
    
    def __init__(self):
        # Common stop words for basic filtering (you can expand this)
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        # Patterns for different types of noise
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'phone': re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
            'excessive_whitespace': re.compile(r'\s+'),
            'special_chars': re.compile(r'[^\w\s\.\,\!\?\;\:\-\(\)]'),
            'repeated_chars': re.compile(r'(.)\1{3,}'),
            'page_numbers': re.compile(r'^\s*\d+\s*$'),
            'headers_footers': re.compile(r'^(header|footer|page \d+|\d+ of \d+).*$', re.IGNORECASE),
            'table_artifacts': re.compile(r'\|.*\|'),
            'bullet_points': re.compile(r'^\s*[•·▪▫◦‣⁃]\s*'),
            'numbering': re.compile(r'^\s*\d+[\.\)]\s*')
        }
    
    def clean_text(self, text: str) -> str:
        """Comprehensive text cleaning"""
        if not text or not isinstance(text, str):
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove URLs and emails (preserve context but remove actual links)
        text = self.patterns['url'].sub('[URL]', text)
        text = self.patterns['email'].sub('[EMAIL]', text)
        text = self.patterns['phone'].sub('[PHONE]', text)
        
        # Clean up formatting artifacts
        text = self.patterns['table_artifacts'].sub(' ', text)
        text = self.patterns['repeated_chars'].sub(r'\1\1', text)
        
        # Normalize whitespace
        text = self.patterns['excessive_whitespace'].sub(' ', text)
        
        # Remove excessive special characters but keep punctuation
        text = self.patterns['special_chars'].sub(' ', text)
        
        # Clean up bullet points and numbering (preserve content, remove markers)
        text = self.patterns['bullet_points'].sub('', text)
        text = self.patterns['numbering'].sub('', text)
        
        return text.strip()
    
    def is_meaningful_content(self, text: str, min_length: int = 20) -> bool:
        """Filter out non-meaningful content"""
        if not text or len(text.strip()) < min_length:
            return False
        
        # Skip page numbers, headers, footers
        if self.patterns['page_numbers'].match(text.strip()):
            return False
        
        if self.patterns['headers_footers'].match(text.strip()):
            return False
        
        # Check if text is mostly special characters
        clean_text = re.sub(r'[^\w\s]', '', text)
        if len(clean_text.strip()) < len(text.strip()) * 0.3:
            return False
        
        # Check for minimum word count
        words = clean_text.split()
        if len(words) < 3:
            return False
        
        return True
    
    def extract_metadata(self, text: str, source: str) -> dict:
        """Extract useful metadata from text"""
        metadata = {'source': source}
        
        # Count sentences and words
        sentences = re.split(r'[.!?]+', text)
        metadata['sentence_count'] = len([s for s in sentences if s.strip()])
        metadata['word_count'] = len(text.split())
        
        # Detect language hints (basic detection)
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if hindi_chars > english_chars * 0.3:
            metadata['language_hint'] = 'hindi_mixed'
        elif english_chars > 0:
            metadata['language_hint'] = 'english'
        else:
            metadata['language_hint'] = 'unknown'
        
        # Detect content type hints
        if re.search(r'\b(table|column|row)\b', text.lower()):
            metadata['content_type'] = 'tabular'
        elif re.search(r'\b(chapter|section|paragraph)\b', text.lower()):
            metadata['content_type'] = 'document'
        else:
            metadata['content_type'] = 'general'
        
        return metadata
    
    def preprocess_document(self, doc, source: str):
        """Main preprocessing function for a document"""
        text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Check if content is meaningful
        if not self.is_meaningful_content(cleaned_text):
            return None
        
        # Extract metadata
        enhanced_metadata = self.extract_metadata(cleaned_text, source)
        
        # Merge with existing metadata
        if hasattr(doc, 'metadata') and doc.metadata:
            enhanced_metadata.update(doc.metadata)
        
        # Create new document with cleaned content and enhanced metadata
        from langchain.schema import Document
        return Document(
            page_content=cleaned_text,
            metadata=enhanced_metadata
        )

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    initialize_models()
    load_vectorstore()

@app.get("/")
async def root():
    return {"message": "Document RAG API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vectorstore_loaded": vectorstore is not None,
        "models_initialized": embedding is not None and llm is not None
    }

import pdfplumber
from langchain.schema import Document

def load_document(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        loader = Docx2txtLoader(path)
        return loader.load()
    elif ext == ".pdf":
        docs = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"source": path, "page": i}))
        return docs
    elif ext == ".csv":
        loader = CSVLoader(path)
        return loader.load()
    elif ext == ".txt":
        loader = TextLoader(path, encoding="utf-8")
        return loader.load()
    else:
        raise ValueError(f"Unsupported file type: {path}")

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(files: List[UploadFile] = File(...)):
    """
    Ingest documents with enhanced preprocessing into the vector database
    """
    global vectorstore, embedding
    
    if not embedding:
        raise HTTPException(status_code=500, detail="Embedding model not initialized")
    
    try:
        preprocessor = TextPreprocessor()
        processed_docs = []
        files_processed = 0
        supported_exts = {".docx", ".pdf", ".csv", ".txt"}

        # Create temporary directory for uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in files:
                ext = os.path.splitext(file.filename)[1].lower()
                if ext not in supported_exts:
                    continue  # skip unsupported files

                # Save uploaded file temporarily
                temp_path = os.path.join(temp_dir, file.filename)
                with open(temp_path, "wb") as temp_file:
                    content = await file.read()
                    temp_file.write(content)

                # Load document using the generic loader
                try:
                    loaded_docs = load_document(temp_path)
                    
                    # Preprocess each document
                    for doc in loaded_docs:
                        processed_doc = preprocessor.preprocess_document(doc, file.filename)
                        if processed_doc:  # Only add meaningful content
                            processed_docs.append(processed_doc)
                    
                    files_processed += 1
                except Exception as load_err:
                    print(f"Error loading {file.filename}: {load_err}")
        
        if not processed_docs:
            raise HTTPException(status_code=400, detail="No valid content found after preprocessing")
        
        # Enhanced text splitting with better parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Increased chunk size for better context
            chunk_overlap=100,  # Increased overlap for better continuity
            length_function=len,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
        )
        
        split_docs = text_splitter.split_documents(processed_docs)
        
        # Filter out very small chunks that might not be meaningful
        meaningful_chunks = [
            doc for doc in split_docs 
            if len(doc.page_content.strip()) >= 50 and 
            len(doc.page_content.split()) >= 5
        ]
        
        if not meaningful_chunks:
            raise HTTPException(status_code=400, detail="No meaningful chunks found after processing")
        
        # Create or update vector store
        if vectorstore is None:
            # Create new vectorstore
            vectorstore = Chroma.from_documents(
                meaningful_chunks,
                embedding,
                persist_directory=persist_dir
            )
        else:
            # Add to existing vectorstore
            vectorstore.add_documents(meaningful_chunks)
        
        return IngestResponse(
            message=f"Successfully ingested {files_processed} files with {len(meaningful_chunks)} meaningful chunks",
            files_processed=files_processed
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during ingestion: {str(e)}")

def extract_answer(response):
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the ingested documents with enhanced context preparation
    """
    global vectorstore, llm
    
    if not vectorstore:
        raise HTTPException(
            status_code=400, 
            detail="No documents ingested. Please ingest documents first."
        )
    
    if not llm:
        raise HTTPException(status_code=500, detail="LLM not initialized")
    
    try:
        # Set up retriever with enhanced search
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
                # "fetch_k": 10,  # Fetch more candidates for better selection
            }
        )
        
        # Retrieve relevant documents
        relevant_docs = retriever.invoke(request.query)
        
        # Sort by relevance and prepare context
        contexts = []
        for i, doc in enumerate(relevant_docs):
            context_snippet = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            contexts.append(f"Context {i+1}: {context_snippet}")
        
        context = "\n\n".join(contexts)
        
        # Enhanced prompt with better instructions
        prompt = f"""
        You are a multilingual compliance assistant trained to answer only with verified information from internal policy documents.
        
        ### User Question:
        {request.query}

        ### Context (from internal documents):
        {context}

        ### Instructions:
        - Detect the language of the user question (English, or Hindi) and respond in the same language.
        - ONLY use the provided context. Do not invent or assume missing information.
        - Your response MUST:
            - Be clear, concise (maximum 30 words), and fact-based.
            - Reflect the most relevant part of the source documents.
            - Mention document name(s) and page number(s) under "References".
        - If no answer is available from the context, say: "I couldn't find that information in the provided documents."

        ### Format:
        Answer: <Your answer here>

        References:
        - <DocumentName> (Page X)
        """
        
        # Get response from LLM
        response = llm.invoke(prompt)
        answer = extract_answer(response)
        
        return QueryResponse(answer=answer)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during query: {str(e)}")

@app.delete("/vectorstore")
async def clear_vectorstore():
    """Clear the vector database"""
    global vectorstore
    
    try:
        if os.path.exists(persist_dir):
            import shutil
            shutil.rmtree(persist_dir)
        
        vectorstore = None
        return {"message": "Vector database cleared successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing vectorstore: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
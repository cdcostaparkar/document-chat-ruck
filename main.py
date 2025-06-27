from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
from typing import List
import os
import tempfile
from pathlib import Path

# LangChain imports
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings

# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
# from langchain_community.llms.ollama import Ollama
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
    # context: str

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

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(files: List[UploadFile] = File(...)):
    """
    Ingest DOCX documents into the vector database
    """
    global vectorstore, embedding
    
    if not embedding:
        raise HTTPException(status_code=500, detail="Embedding model not initialized")
    
    try:
        docs = []
        files_processed = 0
        
        # Create temporary directory for uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in files:
                if not file.filename.endswith('.docx'):
                    continue
                
                # Save uploaded file temporarily
                temp_path = os.path.join(temp_dir, file.filename)
                with open(temp_path, "wb") as temp_file:
                    content = await file.read()
                    temp_file.write(content)
                
                # Load document
                loader = Docx2txtLoader(temp_path)
                docs.extend(loader.load())
                files_processed += 1
        
        if not docs:
            raise HTTPException(status_code=400, detail="No valid DOCX files found")
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50
        )
        split_docs = text_splitter.split_documents(docs)
        
        # Create or update vector store
        if vectorstore is None:
            # Create new vectorstore
            vectorstore = Chroma.from_documents(
                split_docs,
                embedding,
                persist_directory=persist_dir
            )
        else:
            # Add to existing vectorstore
            vectorstore.add_documents(split_docs)
        
        # Persist the vectorstore
        # vectorstore.persist()
        
        return IngestResponse(
            message=f"Successfully ingested {files_processed} documents",
            files_processed=files_processed
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during ingestion: {str(e)}")

def extract_answer(response):
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the ingested documents
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
        # Set up retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Retrieve relevant documents
        relevant_docs = retriever.invoke(request.query)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Prepare prompt
        prompt = f"""Context:\n{context}\n
Question: {request.query}\n
Instructions:
- If the user asks in Hinglish (Hindi-English mix), answer in Hinglish.
- If the user asks in English, answer in English.
- If the user asks in Hindi, answer in Hindi.
- Be concise and stick to the facts (around 30 words or less).
- Use natural, conversational tone appropriate for the language.

Answer: """
        
        # Get response from LLM
        response = llm.invoke(prompt)
        answer = extract_answer(response)
        return QueryResponse(

            answer=answer,
            # context=context
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during query: {str(e)}")

@app.delete("/vectorstore")
async def clear_vectorstore():
    """
    Clear the vector database
    """
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
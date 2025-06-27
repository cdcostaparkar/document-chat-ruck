## 🚀 Setup Instructions

### 1. Clone the Repository

### 2. Create and Activate a Virtual Environment

**On Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install and Run Ollama

> **Ollama is required for local LLM and embedding models.**

- **Install Ollama:**  
  [Follow the official Ollama installation guide](https://ollama.com/download) for your OS.

- **Start Ollama (if not running):**
  ```bash
  ollama serve
  ```

- **Download required models:**
  ```bash
  ollama pull qwen3:0.6b
  ollama pull nomic-embed-text
  ```

### 5. Run the FastAPI Server (FastAPI CLI)

```bash
fastapi dev main.py
```

- The API will be available at: [http://localhost:8000](http://localhost:8000)

## 📝 Notes

- Make sure `ollama serve` is running and the required models are pulled before starting the FastAPI server.
- For document ingestion and querying, use the `/ingest` and `/query` endpoints.

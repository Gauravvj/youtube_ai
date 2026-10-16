# YouTube AI Chatbot (CSE-AIML Mini-Project)

An offline, privacy-first AI Chatbot that ingests YouTube video transcripts and enables conversational Q&A over video content using Retrieval-Augmented Generation (RAG) and local LLMs—without any paid API keys.

---

## 💻 Hardware Constraints & Target Specs
- **RAM**: 8 GB
- **Processor**: Intel Core i5-1240P (12 cores, 16 threads)
- **GPU**: Integrated Intel Iris Xe / UHD Graphics
- **Optimization Strategy**: Lightweight CPU-only inference via `faiss-cpu`, compact embedding model (`all-MiniLM-L6-v2`, ~80MB, 384 dims), and Ollama running `llama3.2:3b` (~2.0 GB VRAM/RAM footprint).

---

## 🛠️ Technology Stack
- **Frontend**: React (TypeScript) + Vite + Tailwind CSS
- **Backend**: Python 3.11 + FastAPI + Uvicorn
- **Transcript Extraction**: `youtube-transcript-api`
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Search**: FAISS (CPU)
- **Local LLM**: Ollama (`llama3.2:3b`)

---

## 📁 Project Structure
```text
youtube_ai/
├── backend/
│   ├── app/
│   │   ├── api/          # API endpoints (transcript, chat, health)
│   │   ├── core/         # Config, logging, constants
│   │   ├── models/       # Pydantic request/response schemas
│   │   ├── services/     # RAG, transcript fetcher, vector store, ollama client
│   │   └── __init__.py
│   └── requirements.txt  # Backend dependencies
├── frontend/             # React + Vite + Tailwind CSS web interface
├── venv/                 # Python 3.11 Virtual Environment
├── .env.example          # Environment variable template
├── .gitignore            # Git exclusion rules
├── requirements.txt      # Root pointer to backend/requirements.txt
└── README.md             # Project documentation
```

---

## 🚀 Setup Instructions

### 1. Prerequisites
- **Python**: 3.11 (configured in `venv/`)
- **Node.js**: v18+ (v24+ installed)
- **Ollama**: Download and install from [ollama.com](https://ollama.com)

### 2. Ollama Local Model Setup
In a terminal, verify Ollama is running and pull the model:
```bash
ollama pull llama3.2:3b
```

### 3. Backend Setup
Activate the Python virtual environment and install dependencies:

**On Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

*(Optional) Create your local `.env` configuration:*
```powershell
Copy-Item .env.example .env
```

### 4. Frontend Setup (Next Phase)
Inside `frontend/`:
```bash
cd frontend
npm install
npm run dev
```

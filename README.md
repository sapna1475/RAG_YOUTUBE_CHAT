# Ask YouTube — Ask Questions About Any YouTube Video

A Chrome extension that lets you ask questions about any YouTube video and get AI-generated answers based on the actual transcript — not just what the model already knows.

Built this because I wanted to understand how RAG pipelines actually work end to end, from raw text all the way to a usable product.

---

## How It Works

```
You open a YouTube video
        │
        ▼
Chrome extension reads the video ID from the URL
        │
        ▼
You type a question and click Ask
        │
        ▼
FastAPI backend fetches the transcript via YouTube Transcript API
        │
        ▼
Transcript is split into overlapping chunks (1000 words, 200 overlap)
        │
        ▼
Each chunk is converted to a 384-dimensional vector embedding
using sentence-transformers/all-MiniLM-L6-v2
        │
        ▼
Vectors are stored in a FAISS index
        │
        ▼
Your question is embedded using the same model
        │
        ▼
FAISS finds the 4 most semantically similar chunks
        │
        ▼
Those chunks are injected as context into a prompt
sent to Qwen 2.5 7B on HuggingFace
        │
        ▼
Answer appears in the extension popup
```

---

## What I Built

| Component | What it does |
|---|---|
| **Chrome Extension** | Reads video ID from YouTube URL, sends question to backend, displays answer |
| **FastAPI Backend** | Handles requests, runs the full RAG pipeline, caches processed videos |
| **RAG Pipeline** | Transcript → chunks → embeddings → FAISS → LLM → answer |
| **Video Cache** | First question per video takes ~15s, follow-up questions take ~2s |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Chrome Extension | Vanilla JS, HTML, CSS |
| Backend | Python, FastAPI, Uvicorn |
| RAG Framework | LangChain |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| LLM | Qwen 2.5 7B Instruct via HuggingFace |
| Transcript | YouTube Transcript API |

---

## Project Structure

```
RAG_YouTube_Chat/
│
├── backend/
│   └── main.py          ← FastAPI server + full RAG pipeline
│
├── extension/
│   ├── manifest.json    ← Chrome extension config
│   ├── content.js       ← runs on YouTube, reads video ID
│   ├── popup.html       ← extension UI
│   └── popup.js         ← handles requests and displays answers
│
├── model.py             ← original RAG prototype (development only)
├── .env                 ← HuggingFace API token
└── requirements.txt
```

---

## Running Locally

### Prerequisites
- Python 3.10+
- Chrome browser
- HuggingFace account (free)

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/RAG_YouTube_Chat.git
cd RAG_YouTube_Chat
```

### 2. Set up virtual environment
```bash
python -m venv venv
.\venv\Scripts\activate      # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install fastapi uvicorn langchain langchain-huggingface
pip install langchain-community faiss-cpu sentence-transformers
pip install youtube-transcript-api python-dotenv
```

### 4. Add your HuggingFace token
Create a `.env` file:
```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```
Get your free token at: https://huggingface.co/settings/tokens

### 5. Start the backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```

Confirm it's running:
```
http://localhost:8000/health  →  {"status": "ok"}
```

### 6. Load the Chrome extension
1. Open `chrome://extensions`
2. Enable **Developer mode** (top right)
3. Click **Load unpacked**
4. Select the `extension/` folder

### 7. Use it
1. Open any YouTube video with captions
2. Click the **YT Answer** icon in your toolbar
3. Type a question
4. Click **Ask**

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Check server is running |
| GET | `/health` | Health check for extension |
| POST | `/ask` | Send video ID + question, get answer |
| DELETE | `/cache/{video_id}` | Clear cache for one video |
| DELETE | `/cache` | Clear all cached videos |

### Example Request
```json
POST /ask
{
  "video_id": "5KmopXwjXik",
  "question": "What is the main topic of this video?"
}
```

### Example Response
```json
{
  "answer": "The video is about...",
  "video_id": "5KmopXwjXik",
  "cached": false
}
```

---

## Key Design Decisions

**Why RAG instead of just asking the LLM directly?**
LLMs don't know what's in a specific YouTube video. RAG lets you inject the actual transcript content as context at query time — no fine-tuning needed, works on any video instantly.

**Why FAISS instead of a cloud vector database?**
For a locally running tool, FAISS is the right choice. It runs in memory, has zero latency overhead, and requires no external service. The tradeoff is that indexes don't persist across server restarts — which is acceptable for this use case.

**Why chunk with overlap?**
When splitting a transcript into chunks, important information can fall at the boundary between two chunks. A 200-word overlap ensures that context at boundaries isn't lost.

**Why cache the built chain?**
Building the FAISS index for a video takes 15-20 seconds. Caching the chain by video ID means follow-up questions on the same video are answered in 2-3 seconds.

---

## Known Limitations

- Backend must be running locally — not deployed to cloud yet
- Only works on YouTube videos that have English captions enabled
- Vector indexes are lost when the server restarts (no persistence layer yet)
- Currently processes the full transcript on first question — could be optimized

-
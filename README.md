# Ask YouTube — Ask Questions About Any YouTube Video

A Chrome extension that answers questions about any YouTube video using a Retrieval-Augmented Generation (RAG) pipeline. Built to understand how RAG systems work end-to-end — not just wiring one together, but measuring whether it actually retrieves useful information.

---


## Demo

[Ask YouTube Demo](assets/Ask_Youtube_demo.gif)

*(Full quality video: [assets/Ask_Youtube_demo.mp4](assets/Ask_Youtube_demo.mp4))*

## Why I Built This

LLMs don't know what's inside a specific YouTube video — they can't answer questions about content outside their training data. Fine-tuning a model per video is impractical. RAG solves this by retrieving relevant transcript chunks at query time and injecting them into the LLM's context, working on any video instantly with no training required.

But RAG pipelines can fail silently — a retriever can look like it's working while actually pulling irrelevant chunks, causing the LLM to hallucinate or give vague answers. So beyond building the pipeline, I built an evaluation system to measure and improve retrieval quality.

---

## Architecture

```
User opens a YouTube video
        │
        ▼
Chrome extension reads the video ID from the URL
        │
        ▼
User asks a question via the extension popup
        │
        ▼
FastAPI backend receives the request
        │
        ▼
Transcript fetched via YouTube Transcript API
        │
        ▼
Transcript split into overlapping chunks (RecursiveCharacterTextSplitter)
        │
        ▼
Chunks embedded using sentence-transformers/all-MiniLM-L6-v2
        │
        ▼
Embeddings stored in a FAISS vector index
        │
        ▼
Question embedded → FAISS retrieves top-k most similar chunks
        │
        ▼
Retrieved chunks injected into a prompt template
        │
        ▼
Qwen 2.5 7B (via HuggingFace) generates a grounded answer
        │
        ▼
Answer returned to the extension popup
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Chrome Extension | Vanilla JavaScript, HTML, CSS |
| Backend | Python, FastAPI, Uvicorn |
| RAG Framework | LangChain |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| LLM | Qwen 2.5 7B Instruct via HuggingFace Inference API |
| Transcript Source | YouTube Transcript API |
| Evaluation | Manual labeling + RAGAS |
| Deployment | Render (backend) |

---

## Retrieval Evaluation

Building the pipeline is the easy part — knowing whether it actually retrieves useful information is what matters. I built a standalone evaluation harness to measure this quantitatively rather than assuming the pipeline worked.


### Results

| Metric | Score |
|---|---|
| Manual Precision@2 (chunk_size=1000, k=2) | **70%** |
| RAGAS context precision | **79.2%** |
| RAGAS faithfulness | **81.1%** |
| RAGAS answer relevancy | **78.5%** |


### Evaluation Pipeline Design

Built as a reusable 3-stage pipeline rather than a one-off script:

```
evaluation/
├── test_questions.py     ← fixed set of 20 test questions
├── run_retrieval.py      ← runs questions through FAISS, saves results to JSON
├── label_relevance.py    ← interactive tool for manual relevance labeling
├── calculate_metrics.py  ← computes Precision@K from labeled data
└── ragas_eval.py         ← automated cross-validation using RAGAS
```

This design lets me re-run the same pipeline with different `chunk_size` or `k` values and compare results without rewriting code — each experiment saves a separate summary JSON for side-by-side comparison.



---

## Project Structure

```
RAG_YouTube_Chat/
│
├── backend/
│   ├── main.py            ← FastAPI server + RAG pipeline
│   ├── requirements.txt
│   └── runtime.txt
│
├── extension/
│   ├── manifest.json
│   ├── content.js          ← reads video ID from YouTube page
│   ├── popup.html
│   └── popup.js            ← handles requests, displays answers
│
├── evaluation/
│   ├── test_questions.py
│   ├── run_retrieval.py
│   ├── label_relevance.py
│   ├── calculate_metrics.py
│   ├── ragas_eval.py
│   └── ragas_summary.json
│
├── assets/
│   └── Ask_Youtube.gif
│
└── README.md
```

---

## Running Locally

### Prerequisites
- Python 3.11+
- Chrome browser
- Free HuggingFace account

### Backend Setup
```bash
git clone https://github.com/sapna1475/RAG_YOUTUBE_CHAT.git
cd RAG_YOUTUBE_CHAT/backend

python -m venv venv
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # Mac/Linux

pip install -r requirements.txt
```

Create a `.env` file in `backend/`:
```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

Run the server:
```bash
uvicorn main:app --reload --port 8000
```

### Extension Setup
1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked** → select the `extension/` folder
4. Open any YouTube video with captions and click the extension icon

### Running the Evaluation Pipeline
```bash
cd evaluation
python run_retrieval.py       # fetches chunks for test questions
python label_relevance.py     # manually label each chunk (interactive)
python calculate_metrics.py   # get your Precision@K score
python ragas_eval.py          # cross-validate with RAGAS
```

---

## Deployment

The backend is deployed on **Render** (free tier). Deployment required pinning the Python runtime version (`runtime.txt` → Python 3.11.9) and relaxing package version pins in `requirements.txt`, since several exact-pinned dependency versions (`faiss-cpu`, `youtube-transcript-api`) weren't available for Render's build environment.


---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Extension pings this before sending requests |
| POST | `/ask` | Main endpoint — video ID + question → answer |
| DELETE | `/cache/{video_id}` | Clear cache for one video |
| DELETE | `/cache` | Clear all cached videos |


**Why RAG instead of fine-tuning?**
Fine-tuning per video is expensive and impractical. RAG retrieves relevant context dynamically at query time, working on any video instantly with zero training.

**Why measure retrieval quality manually before using RAGAS?**
Starting with manual labeling forced me to explicitly define what "relevant" means for this task, so I could sanity-check RAGAS's automated judgments against my own reasoning rather than trusting a black-box score blindly.

**Why cache the built FAISS index per video?**
Building the index takes 15-20 seconds. Caching means follow-up questions on the same video answer in 2-3 seconds instead of reprocessing every time.

---

## What I'd Improve Next

- Expand the evaluation set from 20 to 50+ questions for statistical robustness
- Add a second independent labeler and measure inter-rater agreement
- Measure Recall@K in addition to Precision@K

- Publish the extension to the Chrome Web Store instead of load-unpacked installation

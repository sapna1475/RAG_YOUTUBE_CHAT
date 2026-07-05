#FastAPI Server that connects chrome extension and LangChain RAG pipeline. 
#FastAPi is the web framework that will handle the requests from the chrome extension and pass them to the model.py for processing. It will then return the response back to the chrome extension.
#uvicorn is the server that runs FastAPI.


#cors middleware allows the chrome extension to talk to FastAPI server(otherwise request will be blocked by the browser)


# FastAPI is the web framework
# uvicorn is the server that runs FastAPI
from fastapi import FastAPI, HTTPException

# CORSMiddleware allows the Chrome extension to talk to this server
# Without this, Chrome will block all requests (security policy)
from fastapi.middleware.cors import CORSMiddleware

# BaseModel is from Pydantic — defines the shape of request/response data
from pydantic import BaseModel

# Your existing RAG pipeline imports
#changes now 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# For caching — so we don't re-process the same video twice
# Key = video_id, Value = the ready-to-use chain
from collections import OrderedDict
import os
from dotenv import load_dotenv

load_dotenv()

 
import torch
torch.set_num_threads(1)

# Create the FastAPI app instance
app = FastAPI(
    title="YouTube RAG Assistant API",
    description="Answers questions about YouTube videos using RAG pipeline",
    version="1.0.0"
)

# Allow Chrome extension to make requests to this server
# Without this CORS policy, browser will block all requests
app.add_middleware(
    CORSMiddleware,
    # "*" means allow requests from ANY origin (including Chrome extensions)
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],   # allow GET, POST, DELETE etc.
    allow_headers=["*"],   # allow any headers
)

#cache-aleardy processed video chains so we dont
#refetch + reembed the same video - bounded with LRU style eviction so memory usage cant grow unbounded as more videos are asked
MAX_CACHED_VIDEOS = 3
video_cache: OrderedDict = OrderedDict()

def add_to_cache(video_id: str, chain):
    if video_id in video_cache:
        video_cache.move_to_end(video_id)
    video_cache[video_id] = chain
    if len(video_cache) > MAX_CACHED_VIDEOS:
        oldest_id, _ = video_cache.popitem(last=False)
        print(f"Evicted {oldest_id} from cache (memory limit)")
 
 
def get_from_cache(video_id: str):
    if video_id in video_cache:
        video_cache.move_to_end(video_id)  # mark as recently used
        return video_cache[video_id]
    return None
 

class AskRequest(BaseModel):
    video_id: str   
    question: str   

class AskResponse(BaseModel):
    answer: str      # The LLM's answer
    video_id: str    # Echo back which video was used
    cached: bool     # Whether the video was already cached or freshly processed


# LLM + PROMPT SETUP


print("Loading LLM...")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.2
)
model = ChatHuggingFace(llm=llm)

# Prompt template — same as your model.py
prompt = PromptTemplate(
    template="""
    You are a helpful assistant that answers questions about YouTube videos.
    Answer ONLY from the provided transcript context.
    If the context is not sufficient, just say you don't know.

    Context: {context}
    Question: {question}

    Answer:
    """,
    input_variables=["context", "question"]
)

# Text splitter — same settings as your model.py
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Embeddings model — initialized once, reused for every video

print("Loading embeddings model...")
#changes
#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

print("Server ready!")



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain_for_video(video_id: str):

    # Step 1: Fetch transcript from YouTube
    print(f"Fetching transcript for video: {video_id}")
    transcript = ""

    try:
        ytt = YouTubeTranscriptApi()
        fetched = ytt.fetch(video_id, languages=["en"])
        # Join all snippet texts into one plain string
        transcript = " ".join(chunk.text for chunk in fetched.snippets)
    except TranscriptsDisabled:
        # Raise HTTP 400 error — bad request (video has no captions)
        raise HTTPException(
            status_code=400,
            detail=f"No captions available for video: {video_id}"
        )
    except Exception as e:
        # Any other error (invalid video ID, network issue etc.)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch transcript: {str(e)}"
        )

    if not transcript:
        raise HTTPException(
            status_code=400,
            detail="Transcript is empty"
        )

    # Step 2: Split transcript into chunks
    chunks = splitter.create_documents([transcript])
    print(f"Created {len(chunks)} chunks")

    # Step 3: Create FAISS vector store from chunks
 
    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"Vector store created with {vector_store.index.ntotal} vectors")

    # Step 4: Create retriever from vector store
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}   # fetch top 4 most relevant chunks
    )

    # Step 5: Build the full RAG chain
    # RunnableParallel runs both branches at the same time:
    #   - 'question' branch: passes the question through unchanged
    #   - 'context' branch: retrieves relevant chunks and formats them
    parallel_chain = RunnableParallel({
        'question': RunnablePassthrough(),
        'context': retriever | RunnableLambda(format_docs)
    })


    chain = parallel_chain | prompt | model | StrOutputParser()

    return chain


# ================================================================
# API ENDPOINTS
# ================================================================

@app.get("/")
def root():
    return {"message": "YouTube RAG Assistant is running!"}


@app.get("/health")
def health():
   
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest):

    video_id = body.video_id.strip()
    question = body.question.strip()

    # Validate inputs
    if not video_id:
        raise HTTPException(status_code=400, detail="video_id cannot be empty")
    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty")

    # Check cache first — if we've already processed this video,
    # skip the expensive transcript fetch + embedding step
    chain = get_from_cache(video_id)
    cached = chain is not None

    if not cached: 
        chain = build_chain_for_video(video_id)

        add_to_cache(video_id, chain)

    #run the chain with the user's que
    answer = chain.invoke(question)


    return AskResponse(
        answer=answer,
        video_id=video_id,
        cached=cached
    )


@app.delete("/cache/{video_id}")
def clear_cache(video_id: str):
    if video_id in video_cache:
        del video_cache[video_id]
        return {"message": f"Cache cleared for video: {video_id}"}
    return {"message": f"No cache found for video: {video_id}"}


@app.delete("/cache")
def clear_all_cache():
    video_cache.clear()
    return {"message": "All cache cleared"}
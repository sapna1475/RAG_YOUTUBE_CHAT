#fetch the transcript - split - build FAISS - run each que through the retriever - save both que and ans into a json file 
import json 
import os 
from youtube_transcript_api import YouTubeTranscriptApi , TranscriptsDisabled
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from test_questions import VIDEO_ID , TEST_QUESTIONS

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
K = 4  #no of retrieved chunks 

EXPERIMENT_NAME = f"k{K}_chunk{CHUNK_SIZE}"
OUTPUT_FILE = f"results_{EXPERIMENT_NAME}.json"

def fetch_transcript(video_id : str) -> str:
         
    transcript = ""
    try:
        ytt = YouTubeTranscriptApi()
        fetched = ytt.fetch(video_id, languages=["en"])
        # Join all snippet texts into one plain string
        transcript = " ".join(chunk.text for chunk in fetched.snippets)
    except TranscriptsDisabled:
        # Raise HTTP 400 error — bad request (video has no captions)
        raise RuntimeError(f"No captions available for video: {video_id}")
    
    except Exception as e:
        # Any other error (invalid video ID, network issue etc.)
        raise RuntimeError(
            status_code=500,
            detail=f"Failed to fetch transcript: {str(e)}"
        )

    if not transcript:
        raise RuntimeError("Transcript is empty" )

    return transcript

def build_retriever(transcript : str, chunk_size: int , chunk_overlap: int, k : int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap 
    )
    chunks = splitter.create_documents([transcript])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)

    #create retriever
    retriever = vector_store.as_retriever(
        search_type ="similarity",
        search_kwargs ={"k" : k}
    )
    return retriever

def run_retrieval_for_all_questions(retriever) -> list:
    results = []
    for item in TEST_QUESTIONS:
        question_id= item["id"]
        question_text = item["question"]

        docs = retriever.invoke(question_text)

        chunks = []
        for i , doc in enumerate(docs):
            chunks.append({
                "chunk_index": i,
                "text": doc.page_content,
                "relevant": None    #filled by label_relevance.py
            })

        results.append({
            "question_id" : question_id,
            "question_text": question_text,
            "chunks":chunks
        })
    return results
    

def main():
    print(f"=== Running retrieval experiment: {EXPERIMENT_NAME} ===\n")
 
    print(f"Fetching transcript for video: {VIDEO_ID}")
    transcript = fetch_transcript(VIDEO_ID)
    print(f"Transcript length: {len(transcript)} characters")

    retriever = build_retriever(
        transcript,
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        k=K
    )

    results = run_retrieval_for_all_questions(retriever)

    #save everything to a JSON file 
    output_data = {
        "experiment_name": EXPERIMENT_NAME,
        "video_id": VIDEO_ID,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "k": K,
        "results": results
    }
 
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
 
    print(f"\n✅ Saved retrieval results to: {OUTPUT_FILE}")
    print("Next step: run label_relevance.py to manually label each chunk.")
 
 
if __name__ == "__main__":
    main()
#automated evaluation using RAGAS
#open source llm-judged evaluation framework
#ragas uses an llm to automatically judge(faithfullness, answer relevancy , context-precision)


import json
import os
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import context_precision, faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

from test_questions import TEST_QUESTIONS

# Must match the file created by run_retrieval.py
INPUT_FILE = "results_k2_chunk1000.json"


def load_retrieval_results(filepath: str) -> dict:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found. Run run_retrieval.py first.")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_llm_and_embeddings():
    """
    RAGAS needs an LLM (to judge relevance/faithfulness) and an
    embeddings model (used internally for answer_relevancy, which
    compares embedding similarity between the question and a
    reverse-engineered question generated from the answer).

    We reuse the same HuggingFace models as the rest of the project
    so everything stays consistent and free to run.
    """
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="text-generation",
        temperature=0.2
    )
    chat_model = ChatHuggingFace(llm=llm)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # RAGAS expects its own wrapper classes around LangChain objects
    ragas_llm = LangchainLLMWrapper(chat_model)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    return chat_model, ragas_llm, ragas_embeddings


def generate_answer(chat_model, question: str, contexts: list) -> str:
    """
    Generates an answer using the same prompt style as your main backend,
    but using the ALREADY-RETRIEVED chunks from run_retrieval.py instead
    of re-running FAISS. This keeps the comparison fair — both your
    manual eval and RAGAS are judging the exact same retrieved chunks.
    """
    context_text = "\n\n".join(contexts)

    prompt = f"""
    You are a helpful assistant that answers questions about YouTube videos.
    Answer ONLY from the provided transcript context.
    If the context is not sufficient, just say you don't know.

    Context: {context_text}
    Question: {question}

    Answer:
    """

    response = chat_model.invoke(prompt)
    # ChatHuggingFace returns a message object — extract the text content
    return response.content if hasattr(response, "content") else str(response)


def build_ragas_dataset(data: dict, chat_model) -> Dataset:
    """
    Builds the dataset RAGAS expects: a list of dicts with
    'question', 'contexts' (list of strings), and 'answer'.

    For each question, we generate a fresh answer using the LLM
    so faithfulness and answer_relevancy have something to judge.
    """
    results = data["results"] if "results" in data else data["result"]

    rows = []
    for item in results:
        question = item.get("question") or item.get("question_text")
        contexts = [chunk["text"] for chunk in item["chunks"]]

        print(f"Generating answer for: {question}")
        answer = generate_answer(chat_model, question, contexts)

        rows.append({
            "question": question,
            "contexts": contexts,
            "answer": answer
        })

    return Dataset.from_list(rows)


def main():
    print("=== RAGAS Automated Evaluation ===\n")

    print(f"Loading retrieval results from: {INPUT_FILE}")
    data = load_retrieval_results(INPUT_FILE)

    print("Setting up LLM and embeddings for RAGAS...")
    chat_model, ragas_llm, ragas_embeddings = setup_llm_and_embeddings()

    print("\nGenerating answers for each question (needed for faithfulness/relevancy)...")
    dataset = build_ragas_dataset(data, chat_model)

    print("\nRunning RAGAS evaluation — this may take a few minutes...\n")
    scores = evaluate(
        dataset,
        metrics=[context_precision, faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    print("\n" + "=" * 70)
    print("RAGAS RESULTS")
    print("=" * 70)
    print(scores)

    # Convert to a plain dict and save for comparison against manual results
    results_df = scores.to_pandas()
    summary = {
        "context_precision": float(results_df["context_precision"].mean()),
        "faithfulness": float(results_df["faithfulness"].mean()),
        "answer_relevancy": float(results_df["answer_relevancy"].mean()),
    }

    with open("ragas_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Saved RAGAS summary to: ragas_summary.json")
    print("\nCompare this context_precision score against your manual")
    print("Precision@K from calculate_metrics.py — they should be in a")
    print("similar range if your manual labeling criteria was consistent.")


if __name__ == "__main__":
    main()

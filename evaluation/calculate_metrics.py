#final step of the evaluation pipeline. 
#Precision@K = (no. of relevant chunks retrieved) / K
#its calculated per question
import json 
import os 

INPUT_FILE = "results_k2_chunk1000.json"

def load_results(filepath: str) -> dict: 
    if not os.path.exists(filepath):
        raise FileNotFoundError(
             f"{filepath} not found. Run run_retrieval.py and label_relevance.py first."
        )
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
        
def calculate_precision_per_question(item: dict) -> float: 
    "Precision@K = relevant chunks / total chunks retrieved(K)"

    chunks =item["chunks"]
    k = len(chunks)

    #count relevant labeled chunks(1)
    relevant_count = sum(1 for chunk in chunks if chunk["relevant"] == 1)


    if(k  == 0):
        return 0.0
    
    return relevant_count / k

#checks if every chunk is labeled or not then warns 
def check_all_labeled(results : list) -> bool : 
    unlabeled_count = 0
    for item in results: 
        for chunk in item["chunks"]:
            if chunk["relevant"] is None : 
                unlabeled_count += 1

    if unlabeled_count > 0: 
        print(f"⚠️  Warning: {unlabeled_count} chunk(s) are still unlabeled.")
        print("   Run label_relevance.py again to finish labeling before")
        print("   trusting these results.\n")
        return False
    return True

def main(): 
    print(f"Loading labeled results from : {INPUT_FILE}\n")
    data = load_results(INPUT_FILE)
    
    results = data["results"] if "results" in data else data["result"]
    experiment_name = data.get("experiment_name" , "unknown")
    k = data.get("k" , "?")
    chunk_size = data.get("chunk_size" , "?")

    all_labeled = check_all_labeled(results)

    print("=" * 70)
    print(f"EXPERIMENT: {experiment_name}  (k={k}, chunk_size={chunk_size})")
    print("=" * 70)

    per_question_scores= []

    for item in results: 
        question_text = item.get("question") or item.get("question_text")
        precision = calculate_precision_per_question(item)
        per_question_scores.append(precision)


        # Show relevant/total for clarity, e.g. "1/2 relevant"
        chunks = item["chunks"]
        relevant_count = sum(1 for c in chunks if c["relevant"] == 1)
        total = len(chunks)
 
        print(f"Q{item['question_id']}: {question_text}")
        print(f"   Precision@{k} = {precision:.2f}  ({relevant_count}/{total} relevant)\n")


        #overall avg across all questions
        overall_precision = sum(per_question_scores) / len(per_question_scores)
        
        print("=" * 70)
        print(f"OVERALL PRECISION@{k}: {overall_precision:.2%}")
        print("=" * 70)

        if not all_labeled: 
            print("\n NOTE: this score is based on incomplete labelling.")

        
        #save the summary file for easy comparison across exp. later
        summary = {
            "experiment_name": experiment_name,
            "k": k,
            "chunk_size": chunk_size,
            "overall_precision": round(overall_precision, 4),
            "per_question_precision": [
               {"question_id": r["question_id"], "precision": round(s, 4)}
                for r, s in zip(results, per_question_scores)
            ]
        }

        summary_file = f"summary_{experiment_name}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
           json.dump(summary, f, indent=2, ensure_ascii=False)
 
        print(f"\n✅ Summary saved to: {summary_file}")
        print("\nRun this same pipeline again with different chunk_size or k")
        print("settings, then compare summary files to see what improves precision.")
 
 
if __name__ == "__main__":
    main()

        
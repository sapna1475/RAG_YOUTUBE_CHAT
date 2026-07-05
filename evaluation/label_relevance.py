#s3- now we evaluate the chunks
#reads the .json file - in termal I decide 1, 0 - ans are saved back in the same .json file overwriting

import json
import os


INPUT_FILE = "results_k2_chunk1000.json"

#loads the json file
def load_results(filepath : str)->dict:
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"{filepath} not found. Run run_retrieval.py first."
        )
    with open(filepath , "r" , encoding="utf-8") as f:
        return json.load(f)


#save the labeled file to the same json file
def save_results(filepath: str, data: dict):
    with open(filepath , "w", encoding="utf-8") as f:
        json.dump(data , f, indent=2, ensure_ascii=False)


#relevance input
def get_valid_label() -> int:
    while True: 
        answer = input("Relevant? (1=yes , 0 = no , s = skip): ").strip().lower()
        if answer == "1":
            return 1
        elif answer == "0":
            return 0
        elif answer == "s":
            return None
        else:
            print("Invalid input. Please type 1, 0 or s.")


#Walks through every question and every chunk, showing the text and label
def label_all_chunks(data: dict):
    results = data["results"] if "results" in data else data["result"]
    total_questions = len(results)

    for q_index , item in enumerate(results , start = 1): 
        question_text = item.get("question") or item.get("question_text")
        chunks = item["chunks"]

        print("\n" + "=" * 70)
        print(f"QUESTION {q_index}/{total_questions}: {question_text}")
        print("=" * 70)

        for chunk in chunks : 
            #skip chunk if already labeled
            if chunk["relevant"] is not None: 
                continue

            print(f"\n--- Chunk {chunk['chunk_index']} ---")
            print(chunk["text"])
            print("-" * 40)

            label = get_valid_label()
            chunk["relevant"] = label

            #saave after every single label 
            save_results(INPUT_FILE , data)

            
    print("\n✅ All chunks labeled and saved!")

def main():
    print(f"Loading results from: {INPUT_FILE}\n")
    data = load_results(INPUT_FILE)

    print("Instructions:")
    print("  For each chunk shown, decide if it actually helps answer the question.")
    print("  Type 1 if relevant, 0 if irrelevant, or s to skip for now.\n")
    input("Press Enter to begin labeling...")
 
    label_all_chunks(data)
 
    print(f"\nLabeled results saved to: {INPUT_FILE}")
    print("Next step: run calculate_metrics.py to see your Precision@K score.")
 
 
if __name__ == "__main__":
    main()
import json
from ollama import chat, ChatResponse
from pathlib import Path
from src.scripts.gab_hate import get_gab_columns

root_dir = Path(__file__).resolve().parent.parent


def get_dataset():
    json_file_path = root_dir / "thesis" / "src" / "dataset" / "cad_dataset.json"
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


# Function to classify a comment
def classify_comment(comment, model):
    prompt = f"The following comment needs to be classified. Does it contain hate speech against people with disabilities? Respond only with 'yes' or 'no'. Do not provide any explanations or generate other text.\nComment: {comment}"
    response: ChatResponse = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    # Extract and print the classification result
    return response["message"]["content"]


def classify_dataset(data, model, entry_name):
    output_file_path = (
        root_dir / "thesis" / "src" / "dataset" / "cad_dataset_classified.json"
    )
    processed_comments = 0
    total_comments = len(data)

    for entry in data:
        comment = entry.get("comment", "")
        result = classify_comment(comment, model)
        entry[entry_name] = result  # Add the result to the entry

        processed_comments += 1
        progress = (processed_comments / total_comments) * 100
        print(f"Progress: {progress:.2f}% ({processed_comments}/{total_comments})")

        if processed_comments % 100 == 0:
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                json.dump(data, output_file, indent=4)
            print(f"Progress saved at {processed_comments} comments.")

    # Final save to ensure all data is written
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=4)

    print(
        f"{model} Classification completed, resut can be found in JSON file {output_file_path}"
    )


if __name__ == "__main__":
    data = get_dataset()
    print("Starting classification with llama model...")
    classify_dataset(data, "llama3.1:8b", "result_llama")

    print("Starting classification with gemma model...")
    classify_dataset(data, "gemma", "result_gemma")

    print("Starting classification with mistral model...")
    classify_dataset(data, "mistral", "result_mistral")
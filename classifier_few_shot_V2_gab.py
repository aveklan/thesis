import json
from ollama import chat, ChatResponse
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent


def get_dataset():
    json_file_path = (
        root_dir / "thesis" / "src" / "dataset" / "gab_dataset_withContext.json"
    )
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


# Function to classify a comment
def classify_comment(comment, model):
    examples = """
        The following comment needs to be classified. Does it contain hate speech against people with disabilities? Respond only with 'yes' or 'no'. Do not provide any explanations or generate other text.
        Example 1:
        Comment: "Ah the PSYOPS antifa crew is back. That’s how I know we are winning."
        Response: no

        Example 2:
        Comment: "They weren't innocents, they weren't protestors, they were Hamas scum who want to obliterate Jews and Israel. Leftwing MSM needs to realise Hamas would indoctrinate all of them if they lived under its governance."
        Response: no

        Example 3:
        Comment: "Must be an old picture.  There are no refugees camps or streets filled with human shit."
        Response: no
        
        Example 4:
        Comment: "The white self hating leftist mental disorder that push's people to vote for these scum bags. A few day's earlier he was boasting about how his ancestors came to the US from Italy for a better life. And we need more 3rd worlders from shit holes."
        Response: yes

        Example 5:
        Comment: "Actually that mentally ill fag Bruce Jenner is simply that, a retarded mentally ill faggott."
        Response: yes
    """
    # Add the actual comment to classify
    prompt = f"{examples}\nComment: {comment}\nResponse:"

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
        root_dir / "thesis" / "src" / "dataset" / "gab_dataset_classified_few_shot.json"
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

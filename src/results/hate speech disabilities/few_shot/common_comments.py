import json
from pathlib import Path
from collections import Counter

# Load the JSON files
root_dir = Path(__file__).resolve().parent.parent
mistral_file_path = root_dir / "few_shot" / "mistral_7b" / "no_comments.json"
gemma_file_path = root_dir / "few_shot" / "gemma_7b" / "no_comments.json"
llama3_8b_file_path = root_dir / "few_shot" / "llama3.1_8b" / "no_comments.json"

# Function to load comments by category
def load_comments(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Load the comments from each JSON file
mistral_data = load_comments(mistral_file_path)
gemma_data = load_comments(gemma_file_path)
llama3_data = load_comments(llama3_8b_file_path)

# Function to count appearances across the three files
def count_comments(category):
    mistral_comments = mistral_data.get(category, {}).get("comment", [])
    gemma_comments = gemma_data.get(category, {}).get("comment", [])
    llama3_comments = llama3_data.get(category, {}).get("comment", [])

    # Combine all comments and count occurrences
    combined_comments = mistral_comments + gemma_comments + llama3_comments
    comment_counts = Counter(combined_comments)

    # Filter comments that appear in at least two files
    filtered_comments = [comment for comment, count in comment_counts.items() if count >= 3]
    return filtered_comments

# Process each category
categories = ["cad", "ethos", "gab"]
filtered_data = {category: {"comment": count_comments(category)} for category in categories}

# Save the filtered data to a new JSON file
output_file_path = root_dir / "few_shot" / "common_no_comments.json"
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(filtered_data, output_file, indent=4, ensure_ascii=False)

print(f"Filtered comments saved to {output_file_path}")

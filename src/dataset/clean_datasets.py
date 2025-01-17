import json
from pathlib import Path
import re

root_dir = Path(__file__).resolve().parent


def clean_comment_text(comment):
    """
    Cleans a comment by replacing [linebreak] with \n and normalizing text.
    """
    # Replace `[linebreak]` with `\n` for proper formatting in JSON
    comment = comment.replace("[linebreak]", "\n").strip()
    # Normalize whitespace
    comment = re.sub(r"[^\S\n]+", " ", comment)
    return comment


def clean_dataset(input_file, output_file):

    # Load dataset
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Correct column name typos
    for entry in data:
        if "annotation_ategory" in entry:
            entry["annotation_category"] = entry.pop("annotation_ategory")

    print("Together: ", len(data))

    # Filtered comments which can be classified only taking into account current content
    data = [
        entry
        for entry in data
        if entry.get("annotation_Context", "") != "PreviousContent"
    ]
    print("Only current content: ", len(data))

    # Remove duplicate comments
    seen_comments = set()
    cleaned_data = []
    for entry in data:
        # Clean the comment text
        comment = entry.get("comment", "").strip()
        if not comment:
            continue

        cleaned_comment = clean_comment_text(comment)

        # Skip duplicate comments
        if cleaned_comment in seen_comments:
            continue
        seen_comments.add(cleaned_comment)

        # Update the entry with the cleaned comment
        entry["comment"] = cleaned_comment
        cleaned_data.append(entry)

    print("Final length: ", len(cleaned_data))

    # Save cleaned dataset
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(cleaned_data, file, indent=4, ensure_ascii=False)

    print(f"Dataset cleaned and saved to {output_file}")


if __name__ == "__main__":
    # Define paths
    input_path = root_dir / "cad_dataset_withContext.json"
    output_path = (
        root_dir / "cleaned_json_datasets" / "cad_dataset_withContext_cleaned.json"
    )

    print("Root dir: ", input_path)

    # Clean dataset
    clean_dataset(input_path, output_path)

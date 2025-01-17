import json
import re
from pathlib import Path

# Define the root directory
root_dir = Path(__file__).resolve().parent


def clean_comment_text(comment):
    """
    Cleans a comment by replacing [linebreak] with \n, removing unwanted characters,
    and normalizing whitespace.
    """
    # Replace [linebreak] with \n
    comment = comment.replace("[linebreak]", "\n")
    # Remove invisible characters (e.g., \u200f, \u202e, \uFEFF)
    comment = re.sub(r"[\u200f\u202e\u200F\uFEFF\u2060]", "", comment)
    # Normalize whitespace, preserving newlines
    comment = re.sub(r"[^\S\n]+", " ", comment).strip()
    return comment


def clean_dataset(input_path, output_path, dataset_name=None):
    """
    Cleans the dataset by applying text normalization and removing invalid entries.
    """
    # Load dataset
    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"Processing {dataset_name or 'dataset'}...")
    print("Initial comments: ", len(data))

    # Specific cleaning for CAD dataset
    if dataset_name == "cad":
        for entry in data:
            if "annotation_ategory" in entry:
                entry["annotation_category"] = entry.pop("annotation_ategory")
        # Filter out entries based on context
        data = [
            entry
            for entry in data
            if entry.get("annotation_Context", "") != "PreviousContent"
        ]
        print("Filtered (only current content): ", len(data))

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
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(cleaned_data, file, indent=4, ensure_ascii=False)

    print(f"Cleaned dataset saved to {output_path}")


if __name__ == "__main__":
    # Define input and output paths for all datasets
    datasets = {
        "cad": {
            "input": root_dir / "cad_dataset_withContext.json",
            "output": root_dir
            / "cleaned_json_datasets"
            / "cad_dataset_withContext_cleaned.json",
        },
        "gab": {
            "input": root_dir / "gab_dataset_withContext.json",
            "output": root_dir
            / "cleaned_json_datasets"
            / "gab_dataset_withContext_cleaned.json",
        },
        "ethos": {
            "input": root_dir / "ethos_dataset_withContext.json",
            "output": root_dir
            / "cleaned_json_datasets"
            / "ethos_dataset_withContext_cleaned.json",
        },
    }

    # Ensure output directory exists
    (root_dir / "cleaned_json_datasets").mkdir(exist_ok=True)

    # Clean each dataset
    for dataset_name, paths in datasets.items():
        clean_dataset(paths["input"], paths["output"], dataset_name=dataset_name)

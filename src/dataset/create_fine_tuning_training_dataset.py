import json
from pathlib import Path

from numpy import negative, positive

root_dir = Path(__file__).resolve().parent


def create_formatted_dataset(input_file, output_file):
    """
    Reads a dataset, processes it, and saves it in the proper JSON format
    for multi-label classification.
    """
    formatted_data = []

    # Load the dataset
    with open(input_file, "r", encoding="utf-8") as file:
        raw_data = json.load(file)

    # Process each entry
    for entry in raw_data:
        # Extract the comment and labels
        classification = entry.get("disability", 0.0)
        if classification >= 0.5:
            classification = "yes"
        else:
            classification = "no"

        formatted_entry = {
            "text": "Classify this comment as hate speech or not: "
            + entry.get("comment", "").strip(),  # Comment text
            "label": classification,
        }
        formatted_data.append(formatted_entry)

    # Save the formatted dataset
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(formatted_data, file, indent=4, ensure_ascii=False)

    print(f"Formatted dataset saved to {output_file}")


if __name__ == "__main__":
    # Define input and output paths
    input_path = Path(
        root_dir / "cleaned_json_datasets" / "ethos_dataset_withContext_cleaned.json"
    )  # Replace with your file path
    output_path = Path(
        root_dir / "fine_tuning_datasets" / "ethos_dataset_withContext_formatted.json"
    )  # Replace with desired output path

    # Create the formatted dataset
    create_formatted_dataset(input_path, output_path)

import json
from collections import Counter
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
json_file_path = root_dir / "zero_shot" / "cad_dataset_zero_shot_classified.json"


def normalize_result(result):
    """Normalize the result into broader categories."""
    result = result.strip().lower()
    if result in {"no", "no."}:
        return "No"
    elif result in {"yes", "yes."}:
        return "Yes"
    else:
        return "Other"


def results_comments_disabilities():

    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Filter entries with annotation_category == "ableness/disability"
    filtered_entries = [
        entry
        for entry in data
        if entry.get("annotation_category") == "ableness/disability"
        and entry.get("annotation_Context") == "CurrentContent"
    ]
    total_entries = len(filtered_entries)

    # Normalize and count the results
    normalized_results = [
        normalize_result(entry.get("result_gemma", "")) for entry in filtered_entries
    ]
    result_gemma_counter = Counter(normalized_results)

    print(
        f"Total entries with 'annotation_category' == 'ableness/disability': {total_entries}"
    )
    for result, count in result_gemma_counter.items():
        percentage = (count / total_entries) * 100 if total_entries > 0 else 0
        print(
            f"'result_gemma': '{result}' -> Count: {count}, Percentage: {percentage:.2f}%"
        )


def main():
    print("Creating Datasets...")
    results_comments_disabilities()


if __name__ == "__main__":
    main()

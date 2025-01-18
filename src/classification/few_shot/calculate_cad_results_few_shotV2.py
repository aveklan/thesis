import json
import math
from collections import Counter
from pathlib import Path

# File paths
root_dir = Path(__file__).resolve().parent.parent
cad_file_path = (
    root_dir
    / "few_shot"
    / "classification_results"
    / "cad_dataset_classified_few_shot.json"
)

# Model result keys
MODEL_KEYS = {
    "llama": "result_llama",
    "mistral": "result_mistral",
    "gemma": "result_gemma",
}

# Constants
YES_VALUES = {"yes", "yes.", "yes,"}
NO_VALUES = {"no", "no.", "no,"}


def calculate_accuracy_percentages(comments, model_key):
    """Calculate the percentage of 'yes', 'no', and 'other' responses."""
    counts = Counter(entry.get(model_key, "").strip().lower() for entry in comments)
    total_comments = len(comments)

    yes_percentage = (
        (counts.get("yes", 0) + counts.get("yes.", 0) + counts.get("yes,", 0))
        / total_comments
        * 100
        if total_comments > 0
        else 0
    )
    no_percentage = (
        (counts.get("no", 0) + counts.get("no.", 0) + counts.get("no,", 0))
        / total_comments
        * 100
        if total_comments > 0
        else 0
    )
    other_percentage = 100 - (yes_percentage + no_percentage)

    return [
        round(yes_percentage, 2),
        round(no_percentage, 2),
        round(other_percentage, 2),
    ]


def calculate_global_accuracy_percentages(comments, model_key):
    """Calculate the global accuracy percentages."""
    correct_count = sum(
        1
        for entry in comments
        if (
            (
                entry.get(model_key, "").strip().lower() in YES_VALUES
                and entry.get("annotation_category", "") == "ableness/disability"
            )
            or (
                entry.get(model_key, "").strip().lower() in NO_VALUES
                and entry.get("annotation_category", "") != "ableness/disability"
            )
        )
    )
    total_comments = len(comments)
    correct_percentage = (
        (correct_count / total_comments * 100) if total_comments > 0 else 0
    )
    wrong_percentage = 100 - correct_percentage

    return [round(correct_percentage, 2), round(wrong_percentage, 2)]


def calculate_global_f1_score(comments, model_key):
    """Calculate the F1 score for the model."""
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for entry in comments:
        entry.get(model_key, "").strip().lower() in YES_VALUES
        result = entry.get(model_key, "").strip().lower()
        disability = entry.get("annotation_category", "")

        if result in YES_VALUES:
            if disability == "ableness/disability":
                true_positive += 1
            else:
                false_positive += 1
        elif result in NO_VALUES and disability == "ableness/disability":
            false_negative += 1

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1_score, 2),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def process_and_print_results(comments, title, calculation_function):
    """Generalized function to calculate and print results."""
    print(f"*********     {title}     **********")
    for model_name, model_key in MODEL_KEYS.items():
        results = calculation_function(comments, model_key)
        print(f"{model_name.capitalize()} Results: {results}")


def cad_results():
    """Main function to calculate results for the cad dataset."""
    with open(cad_file_path, "r", encoding="utf-8") as file:
        comments = json.load(file)

    print("Total current + previous content comments:", len(comments))

    # Filtered comments for disability-related hate speech
    current_content_comments = [
        entry
        for entry in comments
        if entry.get("annotation_context", "") != "PreviousContent"
    ]
    print("Total current content comments:", len(current_content_comments))

    disability_comments = [
        entry
        for entry in current_content_comments
        if entry.get("annotation_category", "") == "ableness/disability"
    ]
    print("Disability comments:", len(disability_comments))

    non_hateful_comments = [
        entry
        for entry in current_content_comments
        if entry.get("annotation_Primary", "") == "Neutral"
    ]
    print("Non hateful comments:", len(non_hateful_comments))

    identity_directed_comments = [
        entry
        for entry in current_content_comments
        if entry.get("annotation_Primary", "") == "IdentityDirectedAbuse"
    ]
    print("Identity directed comments:", len(identity_directed_comments))

    # Accuracy for disability-related hate speech
    process_and_print_results(
        disability_comments,
        "Accuracy hate speech people with disabilities",
        calculate_accuracy_percentages,
    )

    # Global accuracy
    process_and_print_results(
        current_content_comments,
        "Global accuracy",
        calculate_global_accuracy_percentages,
    )

    # F1-score
    process_and_print_results(
        comments,
        "F1-score",
        calculate_global_f1_score,
    )


if __name__ == "__main__":
    cad_results()

from sklearn.metrics import classification_report
import json
from ollama import chat, ChatResponse
from pathlib import Path
import torch

root_dir = Path(__file__).resolve().parent.parent.parent

# Paths
test_file = (
    root_dir / "dataset" / "fine_tuning_datasets" / "combined_testing_dataset.json"
)  # Path to testing dataset


# Function to classify a comment using Ollama
def classify_comment(comment, model):
    """
    Send a comment to Ollama for classification and return the response.
    """
    prompt = f"Respond only with 'yes' or 'no'. Do not provide any explanations or generate other text.\nComment: {comment}"
    response: ChatResponse = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    # Extract and return the classification result
    print(response["message"]["content"].strip().lower())
    return response["message"]["content"].strip().lower()  # Ensure consistent format


def evaluate_model(test_data, model):
    """
    Evaluate the model on the test dataset and compute classification metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_true = []
    y_pred = []

    for idx, item in enumerate(test_data, start=1):
        comment = item["text"]
        # Normalize true label: anything other than "yes" becomes "no"
        true_label = (
            "yes"
            if (
                item["label"].strip().lower() == "yes"
                or item["label"].strip().lower() == "yes."
            )
            else "no"
        )

        # Normalize predicted label: anything other than "yes" becomes "no"
        predicted_label = classify_comment(comment, model).strip().lower()
        predicted_label = "yes" if predicted_label == "yes" else "no"

        y_true.append(true_label)
        y_pred.append(predicted_label)

        # Optional: Show progress
        if idx % 10 == 0 or idx == len(test_data):
            print(f"Processed {idx}/{len(test_data)} comments.")

    # Calculate metrics
    report = classification_report(
        y_true, y_pred, labels=["no", "yes"], target_names=["no", "yes"], digits=4
    )
    return report


# Main script
if __name__ == "__main__":
    # Load test dataset
    with open(test_file, "r") as f:
        test_data = json.load(f)

    # Evaluate the model
    metrics_report = evaluate_model(test_data, model="llama3.1:8b")

    # Print the classification report
    print("Evaluation Metrics:")
    print(metrics_report)

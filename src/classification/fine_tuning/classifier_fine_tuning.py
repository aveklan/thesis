from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from pathlib import Path
import torch
from sklearn.metrics import classification_report
import json
import time

# Path configuration
root_dir = Path(__file__).resolve().parent.parent.parent
model_path = (
    root_dir / "fine_tuning" / "Llama-3.1-8B-sft-lora-fine_tuned"
)  # Path to your fine-tuned model
test_file = (
    root_dir / "dataset" / "fine_tuning_datasets" / "combined_testing_dataset.json"
)  # Path to testing dataset

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Quantization Config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto",
)

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Function to classify a comment
def classify_comment(comment, model, tokenizer, device):
    # Tokenize the input comment
    inputs = tokenizer(
        comment,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=512,
    )

    # Move inputs to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    # Convert predicted class to label (assuming binary classification: 0 = No, 1 = Yes)
    label_map = {0: "no", 1: "yes"}  # Match the label format in your dataset
    return label_map[predicted_class]


# Evaluation function
def evaluate_model(test_data, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_true = []
    y_pred = []

    total_comments = len(test_data)
    start_time = time.time()

    for idx, item in enumerate(test_data, start=1):
        comment = item["text"]
        true_label = item["label"].lower()  # Ensure case consistency
        predicted_label = classify_comment(comment, model, tokenizer, device)

        y_true.append(true_label)
        y_pred.append(predicted_label)

        # Print progress
        if (
            idx % 10 == 0 or idx == total_comments
        ):  # Update every 10 comments or at the end
            elapsed_time = time.time() - start_time
            percentage_done = (idx / total_comments) * 100
            avg_time_per_comment = elapsed_time / idx
            estimated_time_remaining = avg_time_per_comment * (total_comments - idx)
            print(
                f"Progress: {percentage_done:.2f}% ({idx}/{total_comments}), "
                f"Elapsed Time: {elapsed_time:.2f}s, "
                f"Estimated Time Remaining: {estimated_time_remaining:.2f}s"
            )

    # Calculate metrics
    report = classification_report(y_true, y_pred, target_names=["no", "yes"], digits=4)
    return report


# Main script
if __name__ == "__main__":
    # Load test dataset
    with open(test_file, "r") as f:
        test_data = json.load(f)

    # Evaluate the model
    metrics_report = evaluate_model(test_data, model, tokenizer)

    # Print the classification report
    print("Evaluation Metrics:")
    print(metrics_report)

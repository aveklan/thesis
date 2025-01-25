import json
import random
from pathlib import Path

root_dir = Path(__file__).resolve().parent

# Load the dataset
input_file = Path(
    root_dir / "cleaned_json_datasets" / "ethos_dataset_withContext_cleaned.json"
)
training_file = Path(
    root_dir / "fine_tuning_datasets" / "ethos_dataset_withContext_training.json"
)
testing_file = Path(
    root_dir / "fine_tuning_datasets" / "ethos_dataset_withContext_testing.json"
)

# Load the JSON file
with open(input_file, "r") as f:
    data = json.load(f)

# Separate comments by category
disability_comments = [item for item in data if item.get("disability", 0.0) >= 0.5]
other_comments = [item for item in data if item.get("disability", 0.0) < 0.5]

# Balance the dataset
balanced_other_comments = random.sample(other_comments, len(disability_comments))

# Combine and format the dataset
formatted_data = []

# Process disability-related comments
for item in disability_comments:
    formatted_data.append(
        {
            "text": f"The following comment needs to be classified. Does it contain hate speech against people with disabilities? {item['comment']}",
            "label": "yes",
        }
    )

# Process other comments
for item in balanced_other_comments:
    formatted_data.append(
        {
            "text": f"The following comment needs to be classified. Does it contain hate speech against people with disabilities? {item['comment']}",
            "label": "no",
        }
    )

# Shuffle the dataset
random.shuffle(formatted_data)

# Split the dataset (70% training, 30% testing)
split_index = int(0.7 * len(formatted_data))
training_data = formatted_data[:split_index]
testing_data = formatted_data[split_index:]

print(len(disability_comments))

# Save the datasets to JSON files
with open(training_file, "w") as train_file:
    json.dump(training_data, train_file, indent=4)

with open(testing_file, "w") as test_file:
    json.dump(testing_data, test_file, indent=4)

print(f"Training dataset saved to {training_file}")
print(f"Testing dataset saved to {testing_file}")

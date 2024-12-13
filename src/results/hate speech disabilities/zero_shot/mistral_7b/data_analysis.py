import json
from pathlib import Path


# Load the JSON file
root_dir = Path(__file__).resolve().parent.parent
json_file_path = root_dir / "mistral_7b" / "gab_classification_results.json"
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Initialize counters
yes_count = 0
no_count = 0
other_count = 0

# Iterate through the JSON data
for entry in data:
    classification = entry.get("classification", "").strip().lower()  # Normalize case
    if classification == "yes":
        yes_count += 1
    elif classification == "no":
        no_count += 1
    else:
        other_count += 1

# Calculate percentages
total = len(data)
yes_percentage = (yes_count / total) * 100 if total > 0 else 0
no_percentage = (no_count / total) * 100 if total > 0 else 0
other_percentage = (other_count / total) * 100 if total > 0 else 0

# Print results
print(f"Total comments: {total}")
print(f"Yes: {yes_percentage:.2f}% ({yes_count})")
print(f"No: {no_percentage:.2f}% ({no_count})")
print(f"Other: {other_percentage:.2f}% ({other_count})")

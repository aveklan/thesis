import json
from pathlib import Path

# Load the JSON file
root_dir = Path(__file__).resolve().parent.parent
json_file_path = root_dir / "mistral_7b" / "gab_classification_results.json"
output_file_path = root_dir / "mistral_7b" / "no_comments.json"
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Read the existing output JSON file, if it exists
if output_file_path.exists():
    with open(output_file_path, 'r', encoding='utf-8') as outfile:
        existing_data = json.load(outfile)
else:
    existing_data = {"gab": {"comment": []}}  # Default structure if file doesn't exist

# Initialize counters and list for comments classified as 'no'
yes_count = 0
no_count = 0
other_count = 0
no_comments = []  # List to store comments with classification 'no'

# Iterate through the JSON data
for entry in data:
    classification = entry.get("classification", "").strip().lower()  # Normalize case
    comment = entry.get("comment", "")  # Retrieve the comment
    if classification == "yes":
        yes_count += 1
    elif classification == "no":
        no_count += 1
        no_comments.append(comment)  # Add the comment to the list
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

# Prepare the JSON structure
# Update the existing data
existing_data["gab"]["comment"].extend(no_comments)

# Save back to the JSON file
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    json.dump(existing_data, outfile, indent=4, ensure_ascii=False)

print(f"'No' comments have been added to {output_file_path}")
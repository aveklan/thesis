import json
import random
from pathlib import Path

root_dir = Path(__file__).resolve().parent

# File paths
gab_training_file = root_dir / "gab_dataset_withContext_training.json"
cad_training_file = root_dir / "cad_dataset_withContext_training.json"
ethos_training_file = root_dir / "ethos_dataset_withContext_training.json"
output_combined_file = root_dir / "combined_training_dataset.json"

# Load datasets
with open(gab_training_file, "r") as f:
    gab_data = json.load(f)

with open(cad_training_file, "r") as f:
    cad_data = json.load(f)

with open(ethos_training_file, "r") as f:
    ethos_data = json.load(f)

# Combine datasets
combined_data = gab_data + cad_data + ethos_data

# Shuffle the combined dataset
random.shuffle(combined_data)

# Save the combined and shuffled dataset
with open(output_combined_file, "w") as f:
    json.dump(combined_data, f, indent=4)

print(f"Combined and shuffled dataset saved to {output_combined_file}")
print(f"Dataset number of comments: {len(combined_data)}")

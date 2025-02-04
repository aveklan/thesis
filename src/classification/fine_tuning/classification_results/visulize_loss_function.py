import matplotlib.pyplot as plt
import ast  # For safely parsing dictionary-like strings
from pathlib import Path

# File path (Change this if your data is in a file)
root_dir = Path(__file__).resolve().parent
file_path = root_dir / "training_results.txt"  # Replace with your actual file

# Initialize lists to store epochs and loss values
epochs = []
losses = []

# Read and parse the file
with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        try:
            data = ast.literal_eval(line.strip())  # Convert string to dictionary
            epochs.append(data["epoch"])
            losses.append(data["loss"])
        except Exception as e:
            print(f"Skipping invalid line: {line}\nError: {e}")

# Plot the loss function
plt.figure(figsize=(8, 5))
plt.plot(epochs, losses, marker="o", linestyle="-", color="b", label="Loss")

# Labels and title
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

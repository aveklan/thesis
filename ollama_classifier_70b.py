from pathlib import Path
import pandas as pd
from ollama import chat, ChatResponse

# Load CSV file
root_dir = Path(__file__).resolve().parent.parent
file_path = root_dir / "Project" / "src" / "dataset" / "dataset_hate_speech_merged.csv"
df = pd.read_csv(file_path)

# Limit to the first 5 rows for testing
df = df[:100]


# Function to classify a comment
def classify_comment(comment):
    prompt = f"The following comment needs to be classified. Does it contain hate speech against people with disabilities? Respond only with 'yes' or 'no'. Do not provide any explanations or generate other text.\nComment: {comment}"
    response: ChatResponse = chat(model="llama3.1:70b", messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    # Extract and print the classification result
    print(response['message']['content'])


# Iterate over the comments in the DataFrame and classify them
for comment in df['comment']:  # Replace 'comment' with your column name
    classification = classify_comment(comment)
    

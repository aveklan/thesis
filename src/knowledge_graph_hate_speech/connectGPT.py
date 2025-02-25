import pandas as pd
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
import json
import re

tqdm.pandas()

root_dir = Path(__file__).resolve().parent
input_path = root_dir / "hate_speech_KG_dataset_comments_with_common_words.json"
output_path = (
    root_dir
    / "hate_speech_KG_dataset_comments_with_common_words_with_relationships.json"
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)


def load_dataset(dataset_path):
    """
    Loads the given dataset by using pandas library.
    Accepts dataset in json format.
    """
    df = pd.read_json(dataset_path, orient="records")
    return df


def generate_relationships(comment, keywords):
    completion = client.chat.completions.create(
        model="google/gemini-2.0-flash-lite-preview-02-05:free",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in extracting RDF triples (subject-predicate-object) from hateful online comments. "
                    "Your goal is to support hate speech research by identifying relationships between hateful terms.\n\n"
                    "Guidelines:\n"
                    "- Each triple must include at least one of the provided keywords as either the predicate or the object.\n"
                    "- The predicate is preferred be a verb.\n"
                    "- Ensure extracted triples maintain the original meaning and context of the comment.\n"
                    "- Subject and object cannot be the same word in the same RDF triple"
                    "- Return the extracted RDF triples in valid JSON format, structured as a list of { 'subject': '', 'predicate': '', 'object': '' } dictionaries.\n"
                    "- If no valid triples can be extracted from a comment, return an empty list []."
                ),
            },
            {
                "role": "user",
                "content": f"""Task: Extract RDF triples (subject-predicate-object) from the given comment. Each triple must contain at least one of the provided keywords as either the predicate or the object. Return the triples in JSON format.

    Input:
    - Comment: "{comment}"
    - Keywords: "{keywords}"

    """,
            },
        ],
        max_tokens=10000,
    )
    return completion.choices[0].message.content


# Load dataframe
df = load_dataset(input_path)

# Initialize the new column for extracted relationships
df["extracted_relationships"] = None

# Load existing data from the output file (if it exists)
try:
    with open(output_path, "r") as file:
        existing_data = json.load(file)
except FileNotFoundError:
    existing_data = []

# Convert existing data to a DataFrame
existing_df = pd.DataFrame(existing_data)

# Iterate over each row in the DataFrame with progress tracking
for index, row in tqdm(df.iterrows(), total=len(df)):
    # Call the API and get the triples
    triples = generate_relationships(row["preprocessedComments"], row["commonEdges"])
    print(index, row, triples)

    # Try parsing JSON safely
    try:
        # Remove Markdown-style triple backticks and "json" label
        cleaned_triples = re.sub(r"```json|```", "", triples).strip()
        parsed_data = json.loads(cleaned_triples)
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed at index {index}: {e}")
        print(f"Problematic JSON: {cleaned_triples}")
        parsed_data = []  # Use an empty list if JSON parsing fails

    # Update the current row with the extracted relationships
    df.at[index, "extracted_relationships"] = parsed_data

    # Append the updated row to the existing data
    updated_row = row.to_dict()
    updated_row["extracted_relationships"] = parsed_data
    existing_data.append(updated_row)

    # Save the updated data to the output file
    with open(output_path, "w") as file:
        json.dump(existing_data, file, indent=4)

print("DataFrame updated and saved incrementally to output.json")

import pandas as pd
import json
from ollama import chat, ChatResponse
from pathlib import Path
from src.scripts.gab_hate import get_gab_columns
from src.scripts.ethos import get_ethos_columns
from src.scripts.cad import get_cad_columns

def get_dataset():
    root_dir = Path(__file__).resolve().parent.parent
    file_path = root_dir / "Project" / "src" / "dataset" / "dataset_hate_speech_merged.csv"
    df = pd.read_csv(file_path)
    return(df[:3])

def getGab():
    df = get_gab_columns()
    return(df)        #276 columns in total

def getEthos():
    df = get_ethos_columns()
    return(df)          #63 columns in total

def getCad():
    df = get_cad_columns()
    return(df)          #442 columns in total

# Function to classify a comment
def classify_comment(comment):
    prompt = f"The following comment needs to be classified. Does it contain hate speech against people with disabilities? Respond only with 'yes' or 'no'. Do not provide any explanations or generate other text.\nComment: {comment}"
    response: ChatResponse = chat(model="gemma", messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    # Extract and print the classification result
    return response['message']['content']


if __name__ == "__main__": 
    df = getEthos()
    results = []
    json_file_path = "ethos_classification_results.json" 
    counter = 0

    for comment in df['comment']:  # Replace 'comment' with your column name
        print('Processing', counter, ":", df.apply(len)[0])
        classification = classify_comment(comment)
        results.append({
            'comment': comment,
            'classification': classification
        })
        counter += 1

    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)
    print(f"Results saved to {json_file_path}")

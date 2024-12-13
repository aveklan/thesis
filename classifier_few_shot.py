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
def classify_comment_ethos(comment):
    # Few-shot examples
    examples = """
The following comment needs to be classified. Does it contain hate speech against people with disabilities? Respond only with 'yes' or 'no'. Do not provide any explanations or generate other text.

Example 1:
Comment: "People with disabilities should be treated equally."
Response: no

Example 2:
Comment: "You look like Sloth with deeper Down syndrome"
Response: yes

Example 3:
Comment: "I think more buildings should be accessible to everyone."
Response: no

Example 2:
Comment: "Throwing stones to paraplegic kids is my favourite hobby"
Response: yes
"""
    # Add the actual comment to classify
    prompt = f"{examples}\nComment: {comment}\nResponse:"
    
    # Send the prompt to the model
    response: ChatResponse = chat(model="gemma", messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    
    # Extract and print the classification result
    return response['message']['content'].strip()

# Function to classify a comment
def classify_comment_gab(comment):
    # Few-shot examples
    examples = """
The following comment needs to be classified. Does it contain hate speech against people with disabilities? Respond only with 'yes' or 'no'. Do not provide any explanations or generate other text.

Example 1:
Comment: "People with disabilities should be treated equally."
Response: no

Example 2:
Comment: "Maybe some people think my language on this is crass, but it's just more political correctness. It's yet another example of getting demonized because you notice patterns. Every mass shooter of every major incident has been autistic or somewhere on the spectrum."
Response: yes

Example 3:
Comment: "I think more buildings should be accessible to everyone."
Response: no

Example 2:
Comment: "Lol you are so mad you can't even type. What a mentally ill curry nigger you are."
Response: yes
"""
    # Add the actual comment to classify
    prompt = f"{examples}\nComment: {comment}\nResponse:"
    
    # Send the prompt to the model
    response: ChatResponse = chat(model="gemma", messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    
    # Extract and print the classification result
    return response['message']['content'].strip()

# Function to classify a comment
def classify_comment_cad(comment):
    # Few-shot examples
    examples = """
The following comment needs to be classified. Does it contain hate speech against people with disabilities? Respond only with 'yes' or 'no'. Do not provide any explanations or generate other text.

Example 1:
Comment: "People with disabilities should be treated equally."
Response: no

Example 2:
Comment: "People that are born without legs are not people. We need a separate category for them."
Response: yes

Example 3:
Comment: "I think more buildings should be accessible to everyone."
Response: no

Example 2:
Comment: "Is this a dwarf or a child? Hard to tell with that face."
Response: yes
"""
    # Add the actual comment to classify
    prompt = f"{examples}\nComment: {comment}\nResponse:"
    
    # Send the prompt to the model
    response: ChatResponse = chat(model="gemma", messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    
    # Extract and print the classification result
    return response['message']['content'].strip()

if __name__ == "__main__": 
    df = getEthos()
    results = []
    json_file_path = "ethos_classification_results.json" 
    counter = 0

    for comment in df['comment']:  # Replace 'comment' with your column name
        print('Processing', counter, ":", df.apply(len)[0])
        classification = classify_comment_ethos(comment)
        results.append({
            'comment': comment,
            'classification': classification
        })
        counter += 1

    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)
    print(f"Results saved to {json_file_path}")

    df = getGab()
    results = []
    json_file_path = "gab_classification_results.json" 
    counter = 0

    for comment in df['comment']:  # Replace 'comment' with your column name
        print('Processing', counter, ":", df.apply(len)[0])
        classification = classify_comment_ethos(comment)
        results.append({
            'comment': comment,
            'classification': classification
        })
        counter += 1

    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)
    print(f"Results saved to {json_file_path}")

    df = getCad()
    results = []
    json_file_path = "cad_classification_results.json" 
    counter = 0

    for comment in df['comment']:  # Replace 'comment' with your column name
        print('Processing', counter, ":", df.apply(len)[0])
        classification = classify_comment_ethos(comment)
        results.append({
            'comment': comment,
            'classification': classification
        })
        counter += 1

    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)
    print(f"Results saved to {json_file_path}")

import pandas as pd
import spacy
import re
import json
from pathlib import Path
from tqdm import tqdm
from spellchecker import SpellChecker

root_dir = Path(__file__).resolve().parent
uinque_tokens_file_path = root_dir / "unique_tokens_list.json"
input_path_cad = (
    root_dir.parent.parent
    / "dataset"
    / "cleaned_json_datasets"
    / "cad_dataset_withContext_cleaned.json"
)
input_path_ethos = (
    root_dir.parent.parent
    / "dataset"
    / "cleaned_json_datasets"
    / "ethos_dataset_withContext_cleaned.json"
)
input_path_gab = (
    root_dir.parent.parent
    / "dataset"
    / "cleaned_json_datasets"
    / "gab_dataset_withContext_cleaned.json"
)

output_path_cad = root_dir / "cad_dataset_withContext_tokenized.json"
output_path_ethos = root_dir / "ethos_dataset_withContext_tokenized.json"
output_path_gab = root_dir / "gab_dataset_withContext_tokenized.json"


nlp = spacy.load("en_core_web_md")
tqdm.pandas()

abbreviation_dict = {
    "gr8": "great",
    "im": "I am",
    "u": "you",
    "r": "are",
    "2": "to",
    "4": "for",
    "b4": "before",
    "l8r": "later",
    "thx": "thanks",
    "plz": "please",
    "omg": "oh my god",
    "btw": "by the way",
    "idk": "I don't know",
    "tbh": "to be honest",
    "irl": "in real life",
    "afaik": "as far as I know",
    "imo": "in my opinion",
    "smh": "shaking my head",
    "ttyl": "talk to you later",
    "brb": "be right back",
    "np": "no problem",
    "jk": "just kidding",
    "lol": "laugh out loud",
    "rofl": "rolling on the floor laughing",
    "gtg": "got to go",
    "wyd": "what are you doing",
    "hbu": "how about you",
    "fyi": "for your information",
    "nvm": "never mind",
    "ofc": "of course",
    "wbu": "what about you",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "kinda": "kind of",
    "sorta": "sort of",
    "outta": "out of",
    "lemme": "let me",
    "gimme": "give me",
    "ain't": "am not",
    "y'all": "you all",
    "fck": "fuck",
    "iq": "intelligence quotient",
    "fk": "fuck",
}
pattern = re.compile(
    r"\b(" + "|".join(re.escape(key) for key in abbreviation_dict.keys()) + r")\b",
    re.IGNORECASE,
)


def remove_mentions(comment):
    text = re.sub(r"@\w+", "you", comment.lower())
    return text


def replace_abbreviations(comment):
    def replace_match(match):
        # Get the matched slang term
        slang = match.group(
            0
        ).lower()  # Convert to lowercase to handle case insensitivity
        # Return the corresponding value from the dictionary
        return abbreviation_dict.get(
            slang, slang
        )  # If slang not found, return the original term

    # Use re.sub() to replace all matches in the comment
    return pattern.sub(replace_match, comment)


# Define the function
def correct_grammar(comment):
    """
    Corrects grammar errors in a comment using TextBlob.
    Args:
        comment (str): The input comment/text.
    Returns:
        str: The comment with grammar errors corrected.
    """
    spell = SpellChecker()
    corrected_text = []
    words = comment.split()
    for word in words:
        # Check if the word is misspelled
        if word in spell:
            corrected_text.append(word)
        else:
            # Suggest the most probable correction
            corrected_word = spell.correction(word)
            corrected_text.append(corrected_word)

    corrected_text = [word if word is not None else "" for word in corrected_text]
    return " ".join(corrected_text)


def preprocess_text(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]


def find_common_tokens(tokenized_text, unique_tokens_high_frequency):
    return list(set(tokenized_text) & set(unique_tokens_high_frequency))


df = pd.read_json(input_path_gab, orient="records")

print("Removing mentions from comments......")
comments = df["comment"]

preprocessed_comments = comments.astype(str).progress_apply(remove_mentions)

print("Replacing abbreviations with grammarly corrected terms......")
preprocessed_comments = preprocessed_comments.astype(str).progress_apply(
    replace_abbreviations
)

# print("Correcting grammar errors...")
# preprocessed_comments = preprocessed_comments.astype(str).progress_apply(
#     correct_grammar
# )

# Load the list from the JSON file
with open(uinque_tokens_file_path, "r", encoding="utf-8") as file:
    unique_tokens_high_frequency = json.load(file)

print("Tokenizing the comments......")
# Apply `preprocess_text()` to each comment and store in a new column
df["tokenized_text"] = preprocessed_comments.astype(str).progress_apply(preprocess_text)

print("Finding common tokens in each comment......")
df["common_tokens"] = df["tokenized_text"].progress_apply(
    lambda x: find_common_tokens(x, unique_tokens_high_frequency)
)

print("Saving the result......")
df.to_json(output_path_gab, orient="records", force_ascii=False)

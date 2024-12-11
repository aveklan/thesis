import pandas as pd
from pathlib import Path
import pandas as pd

def get_gab_columns():
    # Load CSV file
    root_dir = Path(__file__).resolve().parent.parent
    file_path = root_dir / "dataset" / "GabHateCorpus_annotations.tsv"
    dfGab = pd.read_csv(file_path, delimiter='\t')

    dfGab_filtered = dfGab[['ID', 'Annotator', 'Text', 'Hate', 'MPH']]
    dfGab_filtered = dfGab_filtered[dfGab_filtered['MPH'] == 1]
    dfGab_unique = dfGab_filtered.drop_duplicates(subset='Text', keep='first')
    print("Shape of gab hate:", dfGab_unique.shape)

    return dfGab_unique[['Text']].rename(columns={'Text': 'comment'})

if __name__ == "__main__": 
    get_gab_columns()
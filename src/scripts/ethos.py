import pandas as pd
from pathlib import Path
import pandas as pd

def get_columns():
    # Load CSV file
    root_dir = Path(__file__).resolve().parent.parent
    file_path = root_dir / "dataset" / "Ethos_Dataset_Multi_Label.csv"
    dfEthos = pd.read_csv(file_path, delimiter=';')

    dfEthos_filtered = dfEthos[['comment', 'disability']]
    dfEthos_filtered = dfEthos_filtered[dfEthos_filtered['disability'] > 0]
    dfEthos_unique = dfEthos_filtered.drop_duplicates(subset='comment', keep='first')
    print("Shape of df_unique:", dfEthos_unique.shape)

if __name__ == "__main__": 
    get_columns()
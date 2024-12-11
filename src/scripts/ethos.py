import pandas as pd
from pathlib import Path
import pandas as pd


def getDisability (df):
    dfEthos_filtered = df[['comment', 'disability']]
    dfEthos_filtered = dfEthos_filtered[dfEthos_filtered['disability'] > 0]
    dfEthos_unique = dfEthos_filtered.drop_duplicates(subset='comment', keep='first')
    print("Shape of ethos disability:", dfEthos_unique.shape)

    return dfEthos_unique[['comment']]        

def get_ethos_columns():
    # Load CSV file
    root_dir = Path(__file__).resolve().parent.parent
    file_path = root_dir / "dataset" / "Ethos_Dataset_Multi_Label.csv"
    dfEthos = pd.read_csv(file_path, delimiter=';')

    ethos_disability = getDisability(dfEthos)
    return ethos_disability

   

if __name__ == "__main__": 
    get_ethos_columns()
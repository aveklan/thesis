import pandas as pd
from pathlib import Path
import pandas as pd

def get_columns():
    # Load CSV file
    root_dir = Path(__file__).resolve().parent.parent
    file_path = root_dir / "dataset" / "cad_v1.tsv"
    dfCAD = pd.read_csv(file_path, delimiter='\t')

    dfCAD_filtered = dfCAD[['meta_text', 'annotation_Target_top.level.category']]
    dfCAD_filtered = dfCAD_filtered.rename(columns={'annotation_Target_top.level.category': 'annotation'})
    dfCAD_filtered = dfCAD_filtered[dfCAD_filtered['annotation'] == 'ableness/disability']

    dfCAD_unique = dfCAD_filtered.drop_duplicates(subset='meta_text', keep='first')
    print("Shape of df_unique:", dfCAD_unique.shape)

if __name__ == "__main__": 
    get_columns()
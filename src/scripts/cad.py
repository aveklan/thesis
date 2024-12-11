import pandas as pd
file_path = '/content/drive/MyDrive/Dataset Tesi/CAD the Contextual Abuse Dataset/cad_v1.tsv'
dfCAD = pd.read_csv(file_path, delimiter='\t')
print(dfCAD.head())
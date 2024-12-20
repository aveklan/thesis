from math import nan
import pandas as pd
from pathlib import Path
import pandas as pd
import json

root_dir = Path(__file__).resolve().parent.parent


def cad():
    print("Creating CAD dataset...")
    file_path = root_dir / "dataset" / "cad_v1.tsv"
    dfCAD = pd.read_csv(file_path, delimiter="\t")
    dfCAD_filtered = dfCAD[
        [
            "meta_text",
            "annotation_Primary",
            "annotation_Secondary",
            "annotation_Target_top.level.category",
            "annotation_Context",
        ]
    ]
    dfCAD_filtered = dfCAD_filtered.rename(
        columns={"annotation_Target_top.level.category": "annotation_category"}
    )
    dfCAD_filtered = dfCAD_filtered[dfCAD_filtered["meta_text"].notna()]

    dfCAD_filtered_identityDirected = dfCAD_filtered[
        dfCAD_filtered["annotation_Primary"] == "IdentityDirectedAbuse"
    ]
    n = len(dfCAD_filtered_identityDirected)

    dfCAD_filtered_personDirectedAbuse = dfCAD_filtered[
        dfCAD_filtered["annotation_Primary"] == "PersonDirectedAbuse"
    ].head(n)
    dfCAD_filtered_nonHate = dfCAD_filtered[
        dfCAD_filtered["annotation_Primary"] == "Neutral"
    ].head(n)

    cad_dataset = pd.concat(
        [
            dfCAD_filtered_identityDirected,
            dfCAD_filtered_personDirectedAbuse,
            dfCAD_filtered_nonHate,
        ]
    )
    return cad_dataset


def create_dataset(dataset):
    filtered_data = []
    json_file_path = root_dir / "dataset" / "cad_dataset_withContext.json"

    for _, row in dataset.iterrows():
        # Create the JSON structure for each row
        entry = {
            "comment": row["meta_text"],
            "annotation_Primary": row["annotation_Primary"],
            "annotation_ategory": row["annotation_category"],
            "annotation_Context": row["annotation_Context"],
        }
        # Append the entry to the list
        filtered_data.append(entry)

    # Save the list of structured data to a JSON file
    with open(json_file_path, "w", encoding="utf-8") as file:
        json.dump(filtered_data, file, indent=4, ensure_ascii=False)

    print(f"Data successfully saved to {json_file_path}")


def main():
    print("Creating Datasets...")
    identityDirected = cad()
    create_dataset(identityDirected)


if __name__ == "__main__":
    main()

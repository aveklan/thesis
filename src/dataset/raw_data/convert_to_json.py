import pandas as pd
from pathlib import Path
import pandas as pd
import json

root_dir = Path(__file__).resolve().parent.parent


def cad():
    print("Creating CAD dataset...")
    file_path = root_dir / "raw_data" / "cad_v1.tsv"
    dfCAD = pd.read_csv(file_path, delimiter="\t", encoding="utf-8")
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


def gab():
    print("Creating GAB dataset...")
    file_path = root_dir / "raw_data" / "GabHateCorpus_annotations.tsv"
    dfGAB = pd.read_csv(file_path, delimiter="\t")
    dfGAB_aggregated = dfGAB.groupby("Text", as_index=False).sum(numeric_only=True)
    dfGAB_aggregated["Text"] = dfGAB_aggregated["Text"].str.replace(
        r"https://.*", "", regex=True
    )
    dfGAB_aggregated["Text"] = dfGAB_aggregated["Text"].str.replace(
        r"http://.*", "", regex=True
    )

    dfGAB_aggregated_filtered = dfGAB_aggregated.drop(
        columns=["ID", "Annotator"], errors="ignore"
    )

    return dfGAB_aggregated_filtered


def ethos():
    print("Creating ETHOS dataset...")
    file_path = root_dir / "raw_data" / "Ethos_Dataset_Multi_Label.csv"
    dfETHOS = pd.read_csv(file_path, delimiter=";", encoding="utf-8")

    print(dfETHOS)
    return dfETHOS


def create_cad_dataset(dataset):
    filtered_data = []
    json_file_path = root_dir / "cad_dataset_withContext.json"

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


def create_gab_dataset(dataset):
    filtered_data = []
    json_file_path = root_dir / "gab_dataset_withContext.json"

    for _, row in dataset.iterrows():
        # Create the JSON structure for each row
        entry = {
            "comment": row["Text"],
            "Hate": row["Hate"],
            "CV": row["CV"],
            "VO": row["VO"],
            "REL": row["REL"],
            "RAE": row["RAE"],
            "SXO": row["SXO"],
            "GEN": row["GEN"],
            "IDL": row["IDL"],
            "NAT": row["NAT"],
            "POL": row["POL"],
            "MPH": row["MPH"],
            "EX": row["EX"],
            "IM": row["IM"],
        }
        # Append the entry to the list
        filtered_data.append(entry)

    # Save the list of structured data to a JSON file
    with open(json_file_path, "w", encoding="utf-8") as file:
        json.dump(filtered_data, file, indent=4, ensure_ascii=False)

    print(f"Data successfully saved to {json_file_path}")


def create_ethos_dataset(dataset):
    filtered_data = []
    json_file_path = root_dir / "ethos_dataset_withContext.json"

    for _, row in dataset.iterrows():
        # Create the JSON structure for each row
        entry = {
            "comment": row["comment"],
            "violence": row["violence"],
            "directed_vs_generalized": row["directed_vs_generalized"],
            "gender": row["gender"],
            "race": row["race"],
            "national_origin": row["national_origin"],
            "disability": row["disability"],
            "religion": row["religion"],
            "sexual_orientation": row["sexual_orientation"],
        }
        # Append the entry to the list
        filtered_data.append(entry)

    # Save the list of structured data to a JSON file
    with open(json_file_path, "w", encoding="utf-8") as file:
        json.dump(filtered_data, file, indent=4, ensure_ascii=False)

    print(f"Data successfully saved to {json_file_path}")


def main():
    print("Creating Datasets...")
    cadDataset = cad()
    create_cad_dataset(cadDataset)

    gabDataset = gab()
    create_gab_dataset(gabDataset)

    ethosDataset = ethos()
    create_ethos_dataset(ethosDataset)


if __name__ == "__main__":
    main()

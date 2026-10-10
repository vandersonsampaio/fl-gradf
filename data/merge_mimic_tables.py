"""
Merges the raw MIMIC-III tables (ADMISSIONS.csv, PATIENTS.csv,
DIAGNOSES_ICD.csv) into a single `merged_data.csv` with the columns that
`src/utils/hospital_splitter.py` expects: age, comorbidities, specialty,
insurance.

Blocked until `data/download_mimic.py` has been run with real PhysioNet
credentials — this script fails with a clear error if the raw CSVs do not exist.

Note: MIMIC-III has no native "specialty" column in the core tables. Here
it is approximated from ADMISSION_LOCATION (a simple heuristic, not
clinically validated) only to fill the "hospital D" filter criterion
in `hospital_splitter.py` — adjust to the study's real needs.
"""

import argparse
import os

import pandas as pd


def merge_mimic_tables(
    raw_dir: str = "data/raw/mimic",
    output_path: str = "data/raw/mimic/merged_data.csv",
) -> pd.DataFrame:
    admissions_path = os.path.join(raw_dir, "ADMISSIONS.csv")
    patients_path = os.path.join(raw_dir, "PATIENTS.csv")
    diagnoses_path = os.path.join(raw_dir, "DIAGNOSES_ICD.csv")

    for path in (admissions_path, patients_path, diagnoses_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run `python data/download_mimic.py` "
                "with real PhysioNet credentials first."
            )

    admissions = pd.read_csv(admissions_path)
    patients = pd.read_csv(patients_path)
    diagnoses = pd.read_csv(diagnoses_path)

    admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"])
    patients["DOB"] = pd.to_datetime(patients["DOB"])

    df = admissions.merge(patients[["SUBJECT_ID", "DOB"]], on="SUBJECT_ID", how="left")

    # Age at admission. MIMIC-III shifts the date of birth of
    # patients older than 89 into the past (anonymization) — this artifact is
    # not corrected here, consistent with the simplification already in
    # hospital_splitter.py.
    df["age"] = (df["ADMITTIME"] - df["DOB"]).dt.days / 365.25

    comorbidities = diagnoses.groupby("HADM_ID").size().rename("comorbidities")
    df = df.merge(comorbidities, on="HADM_ID", how="left")
    df["comorbidities"] = df["comorbidities"].fillna(0).astype(int)

    df["insurance"] = df["INSURANCE"].str.lower()

    # Heuristic proxy for "specialty" — see the note in the module docstring.
    df["specialty"] = df["ADMISSION_LOCATION"].apply(
        lambda loc: "referral" if isinstance(loc, str) and "TRANSFER" in loc.upper() else "general"
    )

    columns = ["SUBJECT_ID", "HADM_ID", "age", "comorbidities", "specialty", "insurance"]
    result = df[columns].dropna(subset=["age"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", default="data/raw/mimic")
    parser.add_argument("--output_path", default="data/raw/mimic/merged_data.csv")
    args = parser.parse_args()

    df = merge_mimic_tables(args.raw_dir, args.output_path)
    print(f"{len(df)} rows saved to {args.output_path}")

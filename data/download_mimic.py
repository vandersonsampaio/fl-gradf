"""
Downloads the MIMIC-III (v1.4) dataset from PhysioNet into data/raw/mimic/.

Requires a credentialed PhysioNet account with MIMIC-III access (includes
completing the CITI training) — see https://physionet.org/content/mimiciii/.
Edit `username` below before running; the password is asked interactively
(it is not visible in `ps`/shell history).

After downloading, run `python data/merge_mimic_tables.py` to build the
`merged_data.csv` that `src/utils/hospital_splitter.py` expects.
"""

import getpass
import glob
import subprocess

username = "your_email@example.com"


if __name__ == "__main__":
    if username == "your_email@example.com":
        raise SystemExit(
            "Edit the `username` variable in this file with your PhysioNet e-mail before running."
        )

    password = getpass.getpass("PhysioNet password: ")

    # -nH --cut-dirs=3 keeps wget from replicating the URL's directory tree
    # (physionet.org/files/mimiciii/1.4/...) inside data/raw/mimic/.
    subprocess.run(
        [
            "wget", "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=3",
            "--user", username, "--password", password,
            "https://physionet.org/files/mimiciii/1.4/",
            "-P", "./data/raw/mimic/",
        ],
        check=True,
    )

    csv_files = glob.glob("data/raw/mimic/**/*.csv", recursive=True)
    print(f"✓ {len(csv_files)} CSV files found in data/raw/mimic/")
    print("Next step: python data/merge_mimic_tables.py")

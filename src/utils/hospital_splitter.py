import os

import pandas as pd
import numpy as np


def create_non_iid_hospitals():
    """Splits MIMIC into 5 hospitals with different distributions"""

    # Load data
    df = pd.read_csv('data/raw/mimic/merged_data.csv')

    # Hospital A: Wealthy urban (elderly, well cared for)
    hosp_a = df[
        (df['age'] >= 60) &
        (df['comorbidities'] >= 3)
        ].sample(n=2000, random_state=42)
    hosp_a['readmission_30day'] = np.where(
        np.random.random(len(hosp_a)) < 0.15, 1, 0
    )  # 15% readmission

    # Hospital B: Middle-class urban
    hosp_b = df[
        (df['age'] >= 40) & (df['age'] < 60) &
        (df['comorbidities'] >= 2)
        ].sample(n=2500, random_state=43)
    hosp_b['readmission_30day'] = np.where(
        np.random.random(len(hosp_b)) < 0.25, 1, 0
    )  # 25% readmission

    # Hospital C: Rural (younger, worse health)
    hosp_c = df[
        (df['age'] < 50) &
        (df['comorbidities'] <= 1)
        ].sample(n=1500, random_state=44)
    hosp_c['readmission_30day'] = np.where(
        np.random.random(len(hosp_c)) < 0.45, 1, 0
    )  # 45% readmission

    # Hospital D: Teaching hospital
    hosp_d = df[
        df['specialty'] == 'referral'
        ].sample(n=3000, random_state=45)
    hosp_d['readmission_30day'] = np.where(
        np.random.random(len(hosp_d)) < 0.35, 1, 0
    )  # 35% readmission

    # Hospital E: Private (wealthy, well managed)
    hosp_e = df[
        (df['age'] >= 50) &
        (df['insurance'] == 'private')
        ].sample(n=1000, random_state=46)
    hosp_e['readmission_30day'] = np.where(
        np.random.random(len(hosp_e)) < 0.12, 1, 0
    )  # 12% readmission

    # Save
    hospitals = {
        'hospital_a': hosp_a,
        'hospital_b': hosp_b,
        'hospital_c': hosp_c,
        'hospital_d': hosp_d,
        'hospital_e': hosp_e,
    }

    for name, data in hospitals.items():
        os.makedirs(f'data/processed/{name}/train/', exist_ok=True)
        os.makedirs(f'data/processed/{name}/test/', exist_ok=True)

        # Split train/test 80/20
        train = data.sample(frac=0.8, random_state=42)
        test = data.drop(train.index)

        train.to_csv(f'data/processed/{name}/train/data.csv', index=False)
        test.to_csv(f'data/processed/{name}/test/data.csv', index=False)

        print(f"✓ {name}: {len(train)} train, {len(test)} test")
        print(f"  Readmission rate: {train['readmission_30day'].mean():.1%}")

    return hospitals


def calculate_non_iid_coefficient(hospitals):
    """Non-IID = 1 - (minimum rate / maximum rate) of readmission across hospitals"""
    rates = [df['readmission_30day'].mean() for df in hospitals.values()]
    return 1 - (min(rates) / max(rates))


if __name__ == '__main__':
    # Requires data/raw/mimic/merged_data.csv — see data/merge_mimic_tables.py
    # and PhysioNet credentials in data/download_mimic.py. Blocked until then.
    hospitals = create_non_iid_hospitals()
    non_iid = calculate_non_iid_coefficient(hospitals)
    print(f"Non-IID coefficient: {non_iid:.2f}")
import pandas as pd
from sklearn.model_selection import train_test_split


# -------------------------------
# 1. Load Data
# -------------------------------
def load_data():
    beneficiary = pd.read_csv("data/Train_Beneficiarydata.csv")
    inpatient = pd.read_csv("data/Train_Inpatientdata.csv")
    outpatient = pd.read_csv("data/Train_Outpatientdata.csv")
    provider = pd.read_csv("data/Train_Provider.csv")

    return beneficiary, inpatient, outpatient, provider


# -------------------------------
# 2. Merge Data
# -------------------------------
def merge_data(beneficiary, inpatient, outpatient, provider):

    # Merge inpatient with beneficiary
    inp = pd.merge(inpatient, beneficiary, on="BeneID", how="left")

    # Merge outpatient with beneficiary
    outp = pd.merge(outpatient, beneficiary, on="BeneID", how="left")

    # Combine both claim types
    data = pd.concat([inp, outp], axis=0, ignore_index=True)

    # Merge with provider fraud labels
    data = pd.merge(data, provider, on="Provider", how="left")

    return data


# -------------------------------
# 3. Clean Data (FIXED)
# -------------------------------
def clean_data(df):

    # Convert fraud label first
    df['PotentialFraud'] = df['PotentialFraud'].map({'Yes': 1, 'No': 0})

    # -------------------------------
    # Handle numeric columns ONLY
    # -------------------------------
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(0)

    # -------------------------------
    # Handle categorical columns
    # -------------------------------
    cat_cols = df.select_dtypes(include=['object']).columns

    for col in cat_cols:
        if col not in ['AdmissionDt', 'DischargeDt', 'DOB']:
            df[col] = df[col].fillna("Unknown")

    # -------------------------------
    # DO NOT TOUCH DATE COLUMNS HERE
    # (Handled in feature engineering)
    # -------------------------------

    return df


# -------------------------------
# 4. Prepare Data for Model
# -------------------------------
def prepare_data(df):

    features = [
        'ClaimCount',
        'AvgClaimAmount',
        'Age',
        'ChronicCount',
        'StayDuration',
        'ClaimRatio'   # Added new feature
    ]

    # Ensure all features exist
    df = df.copy()
    for col in features:
        if col not in df.columns:
            df[col] = 0

    X = df[features]
    y = df['PotentialFraud']

    return train_test_split(X, y, test_size=0.2, random_state=42)
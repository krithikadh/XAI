import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    beneficiary = pd.read_csv("data/Train_Beneficiarydata.csv")
    inpatient = pd.read_csv("data/Train_Inpatientdata.csv")
    outpatient = pd.read_csv("data/Train_Outpatientdata.csv")
    provider = pd.read_csv("data/Train_Provider.csv")
    return beneficiary, inpatient, outpatient, provider

def merge_data(beneficiary, inpatient, outpatient, provider):
    inp = pd.merge(inpatient, beneficiary, on="BeneID", how="left")
    outp = pd.merge(outpatient, beneficiary, on="BeneID", how="left")
    data = pd.concat([inp, outp], axis=0, ignore_index=True)
    data = pd.merge(data, provider, on="Provider", how="left")
    return data

def clean_data(df):
    df['PotentialFraud'] = df['PotentialFraud'].map({'Yes': 1, 'No': 0})
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(0)
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if col not in ['AdmissionDt', 'DischargeDt', 'DOB']:
            df[col] = df[col].fillna("Unknown")
    return df

def prepare_data(df):
    features = [
        'ClaimCount', 'AvgClaimAmount', 'Age',
        'ChronicCount', 'StayDuration',
        'ClaimRatio', 'HighClaimFlag',
        'FrequentClaimFlag', 'RiskScore'
    ]
    for col in features:
        if col not in df.columns:
            df[col] = 0
    X = df[features]
    y = df['PotentialFraud']
    return train_test_split(X, y, test_size=0.2, random_state=42)
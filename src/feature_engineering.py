import pandas as pd

def create_features(df):

    df['ClaimCount'] = df.groupby('Provider')['ClaimID'].transform('count')
    df['AvgClaimAmount'] = df.groupby('Provider')['InscClaimAmtReimbursed'].transform('mean')

    df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
    df['Age'] = 2024 - df['DOB'].dt.year
    df['Age'] = df['Age'].fillna(df['Age'].median())

    chronic_cols = [col for col in df.columns if 'ChronicCond' in col]
    df['ChronicCount'] = df[chronic_cols].fillna(0).sum(axis=1)

    df['AdmissionDt'] = pd.to_datetime(df['AdmissionDt'], errors='coerce')
    df['DischargeDt'] = pd.to_datetime(df['DischargeDt'], errors='coerce')
    df['StayDuration'] = (df['DischargeDt'] - df['AdmissionDt']).dt.days
    df['StayDuration'] = df['StayDuration'].fillna(0)

    df['ClaimRatio'] = df['InscClaimAmtReimbursed'] / (df['AvgClaimAmount'] + 1)

    df['HighClaimFlag'] = (df['InscClaimAmtReimbursed'] > 10000).astype(int)
    df['FrequentClaimFlag'] = (df['ClaimCount'] > 20).astype(int)

    df['RiskScore'] = (
        df['ClaimCount'] * 0.4 +
        df['AvgClaimAmount'] * 0.0001 +
        df['ChronicCount'] * 0.3 +
        df['StayDuration'] * 0.2
    )

    return df
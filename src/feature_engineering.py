import pandas as pd

def create_features(df):

    # -------------------------------
    # 1. Claim-based features
    # -------------------------------

    # Claim count per provider
    df['ClaimCount'] = df.groupby('Provider')['ClaimID'].transform('count')

    # Average claim amount per provider
    df['AvgClaimAmount'] = df.groupby('Provider')['InscClaimAmtReimbursed'].transform('mean')


    # -------------------------------
    # 2. Patient-based features
    # -------------------------------

    # Convert DOB safely
    df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')

    # Age calculation
    df['Age'] = 2024 - df['DOB'].dt.year
    df['Age'] = df['Age'].fillna(0)


    # -------------------------------
    # 3. Chronic condition features
    # -------------------------------

    chronic_cols = [col for col in df.columns if 'ChronicCond' in col]

    if chronic_cols:
        df[chronic_cols] = df[chronic_cols].fillna(0)
        df['ChronicCount'] = df[chronic_cols].sum(axis=1)
    else:
        df['ChronicCount'] = 0


    # -------------------------------
    # 4. Hospital stay duration (FIXED)
    # -------------------------------

    if 'AdmissionDt' in df.columns and 'DischargeDt' in df.columns:

        # Convert safely (handles "0" and invalid values)
        df['AdmissionDt'] = pd.to_datetime(df['AdmissionDt'], errors='coerce')
        df['DischargeDt'] = pd.to_datetime(df['DischargeDt'], errors='coerce')

        # Calculate duration
        df['StayDuration'] = (df['DischargeDt'] - df['AdmissionDt']).dt.days

        # Replace invalid/missing values
        df['StayDuration'] = df['StayDuration'].fillna(0)

    else:
        df['StayDuration'] = 0


    # -------------------------------
    # 5. Additional useful features (OPTIONAL but good)
    # -------------------------------

    # Claim amount ratio (avoid division by zero)
    df['ClaimRatio'] = df['InscClaimAmtReimbursed'] / (df['AvgClaimAmount'] + 1)


    # -------------------------------
    # 6. Final cleanup
    # -------------------------------

    # Replace any remaining NaNs in numeric columns
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df
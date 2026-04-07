from fastapi import FastAPI, Form #uvicorn app.main:app --reload(to run)
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

app = FastAPI()

# Load models
cat_model = joblib.load("models/catboost.pkl")
iso_model = joblib.load("models/isolation.pkl")

# ------------
# 1. HOMEPAGE
# ------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>XAI Healthcare Fraud Detection</title>
        <style>
            body {
                margin: 0;
                font-family: 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #ffe0dc, #e0c3fc);
                color: #2c2c2c;
            }

            .container {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }

            .card {
                background: rgba(255,255,255,0.85);
                backdrop-filter: blur(12px);
                padding: 40px;
                border-radius: 20px;
                width: 420px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.15);
            }

            h1 {
                text-align: center;
                color: #ff6f61;
                font-size: 28px;
            }

            h2 {
                text-align: center;
                color: #555;
                margin-bottom: 25px;
            }

            label {
                display: block;
                margin-top: 12px;
            }

            input {
                width: 100%;
                padding: 12px;
                margin-top: 5px;
                border-radius: 10px;
                border: 1px solid #ddd;
                background: white;
            }

            input:focus {
                border-color: #ff6f61;
                box-shadow: 0 0 5px rgba(255,111,97,0.3);
            }

            button {
                width: 100%;
                padding: 14px;
                margin-top: 25px;
                border-radius: 12px;
                border: none;
                background: linear-gradient(90deg, #ff6f61, #ff9472);
                color: white;
                font-weight: bold;
                cursor: pointer;
                transition: 0.3s;
            }

            button:hover {
                transform: scale(1.05);
                box-shadow: 0 5px 15px rgba(255,111,97,0.4);
            }
        </style>
    </head>

    <body>
        <div class="container">
            <div class="card">
                <h1>XAI</h1>
                <h2>Healthcare Insurance Fraud Detection</h2>

                <form action="/predict_form" method="post">
                    <label>Claim Count</label>
                    <input type="number" name="ClaimCount" required>

                    <label>Average Claim Amount</label>
                    <input type="number" name="AvgClaimAmount" required>

                    <label>Age</label>
                    <input type="number" name="Age" required>

                    <label>Chronic Condition Count</label>
                    <input type="number" name="ChronicCount" required>

                    <label>Stay Duration</label>
                    <input type="number" name="StayDuration" required>

                    <button type="submit">Analyze Claim</button>
                </form>
            </div>
        </div>
    </body>
    </html>
    """


# -------------------------------
# 2. PREDICTION + RESULT
# -------------------------------
@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(
    ClaimCount: int = Form(...),
    AvgClaimAmount: float = Form(...),
    Age: int = Form(...),
    ChronicCount: int = Form(...),
    StayDuration: int = Form(...)
):

    if any(x < 0 for x in [ClaimCount, AvgClaimAmount, Age, ChronicCount, StayDuration]):
        return "<h3>Invalid input</h3>"

    df = pd.DataFrame([{
        "ClaimCount": ClaimCount,
        "AvgClaimAmount": AvgClaimAmount,
        "Age": Age,
        "ChronicCount": ChronicCount,
        "StayDuration": StayDuration
    }])

    # ---------------- FEATURE ENGINEERING ----------------
    df["ClaimRatio"] = df["AvgClaimAmount"] / (df["AvgClaimAmount"] + 1)
    df["HighClaimFlag"] = (df["AvgClaimAmount"] > 10000).astype(int)
    df["FrequentClaimFlag"] = (df["ClaimCount"] > 20).astype(int)

    df["RiskScore"] = (
        df["ClaimCount"] * 0.4 +
        df["AvgClaimAmount"] * 0.0001 +
        df["ChronicCount"] * 0.3 +
        df["StayDuration"] * 0.2
    )

    df = df[
        [
            'ClaimCount',
            'AvgClaimAmount',
            'Age',
            'ChronicCount',
            'StayDuration',
            'ClaimRatio',
            'HighClaimFlag',
            'FrequentClaimFlag',
            'RiskScore'
        ]
    ]

    # ---------------- PREDICTION ----------------
    prob = cat_model.predict_proba(df)[0][1]
    anomaly = iso_model.predict(df)[0]

    anomaly_flag = 1 if anomaly == -1 else 0

    if ClaimCount > 25:
        fraud = 1
    elif prob > 0.7 and anomaly_flag == 1:
        fraud = 1
    elif prob > 0.8:
        fraud = 1
    elif anomaly_flag == 1 and prob > 0.5:
        fraud = 1
    else:
        fraud = 0

    result = "FRAUD DETECTED 🚨" if fraud else "NOT FRAUD ✅"

    # ---------------- SHAP ----------------
    explainer = shap.Explainer(cat_model)
    shap_values = explainer(df)

    feature_importance = dict(zip(df.columns, shap_values.values[0]))
    top_features = sorted(
        feature_importance.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]

    shap_sentences = []

    for feature, value in top_features:

        if fraud == 1:

            if feature == "ClaimCount":
                shap_sentences.append(
                    "High number of claims increased fraud likelihood"
                )

            elif feature == "AvgClaimAmount":
                shap_sentences.append(
                    "Very high claim amount increased fraud likelihood"
                )

            elif feature == "ChronicCount":
                shap_sentences.append(
                    "Multiple chronic conditions increased fraud likelihood"
                )

            elif feature == "StayDuration":
                shap_sentences.append(
                    "Long hospital stay increased fraud likelihood"
                )

            elif feature == "Age":
                shap_sentences.append(
                    "Patient age contributed to fraud risk"
                )

        else:

            if feature == "ClaimCount":
                shap_sentences.append(
                    "Low claim count reduced fraud likelihood"
                )

            elif feature == "AvgClaimAmount":
                shap_sentences.append(
                    "Lower claim amount reduced fraud likelihood"
                )

            elif feature == "ChronicCount":
                shap_sentences.append(
                    "Fewer chronic conditions reduced fraud likelihood"
                )

            elif feature == "StayDuration":
                shap_sentences.append(
                    "Short hospital stay reduced fraud likelihood"
                )

            elif feature == "Age":
                shap_sentences.append(
                    "Patient age indicates low fraud risk"
                )


    # ---------------- LIME ----------------
    lime_explainer = LimeTabularExplainer(
        training_data=np.array(df),
        feature_names=df.columns.tolist(),
        mode='classification'
    )

    lime_exp = lime_explainer.explain_instance(
        df.iloc[0].values,
        cat_model.predict_proba
    )

    lime_sentences = []

    for rule, _ in lime_exp.as_list()[:3]:

        if fraud == 1:
            lime_sentences.append(rule + " suggests fraud")
        else:
            lime_sentences.append(rule + " suggests safe claim")

    # ---------------- ATTRIBUTE MEANINGS ----------------
    attribute_meaning = """
    <ul>
    <li><b>ClaimCount</b> — Number of claims submitted by provider</li>
    <li><b>AvgClaimAmount</b> — Average reimbursement per claim</li>
    <li><b>ChronicCount</b> — Number of long-term medical conditions</li>
    <li><b>StayDuration</b> — Hospital stay duration in days</li>
    <li><b>Age</b> — Patient age</li>
    </ul>
    """

    # ---------------- RESULT  ----------------
    return f"""
    <html>
    <head>
        <style>
            body {{
                background: linear-gradient(135deg, #ffe0dc, #e0c3fc);
                font-family: 'Segoe UI', sans-serif;
            }}

            .container {{
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }}

            .card {{
                background: rgba(255,255,255,0.9);
                padding: 35px;
                border-radius: 20px;
                width: 480px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }}

            h2 {{
                text-align: center;
                color: #ff6f61;
            }}

            .result {{
                text-align: center;
                font-size: 22px;
                font-weight: bold;
                margin: 15px 0;
                color: {'#d00000' if fraud else '#2d6a4f'};
            }}

            ul {{
                padding-left: 20px;
            }}

            a {{
                display: block;
                text-align: center;
                margin-top: 20px;
                padding: 12px;
                border-radius: 10px;
                background: #ff6f61;
                color: white;
                text-decoration: none;
            }}

            .info {{
                background:#fff4ef;
                padding:15px;
                border-radius:12px;
                margin-top:15px;
            }}
        </style>
    </head>

    <body>
        <div class="container">
            <div class="card">
                <h2>AI Fraud Analysis</h2>

                <div class="result">{result}</div>
                <p style="text-align:center;">Probability: {prob:.2f}</p>

                <h3>SHAP</h3>
                <ul>{''.join(f"<li>{s}</li>" for s in shap_sentences)}</ul>

                <h3>LIME</h3>
                <ul>{''.join(f"<li>{s}</li>" for s in lime_sentences)}</ul>

                <div class="info">
                <h3>Attribute Meaning</h3>
                {attribute_meaning}
                </div>

                <a href="/">Try Another</a>
            </div>
        </div>
    </body>
    </html>
    """
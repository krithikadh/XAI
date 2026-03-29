from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd

app = FastAPI()

# Load models
cat_model = joblib.load("models/catboost.pkl")
iso_model = joblib.load("models/isolation.pkl")


# -------------------------------
# 1. Homepage (UI Form)
# -------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Fraud Detection</title>
        </head>
        <body style="font-family: Arial; padding: 30px;">
            <h2>Healthcare Fraud Detection</h2>

            <form action="/predict_form" method="post">
                Claim Count: <input type="number" name="ClaimCount"><br><br>
                Avg Claim Amount: <input type="number" name="AvgClaimAmount"><br><br>
                Age: <input type="number" name="Age"><br><br>
                Chronic Count: <input type="number" name="ChronicCount"><br><br>
                Stay Duration: <input type="number" name="StayDuration"><br><br>

                <button type="submit">Check Fraud</button>
            </form>
        </body>
    </html>
    """


# -------------------------------
# 2. Prediction from Form
# -------------------------------
@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(
    ClaimCount: int = Form(...),
    AvgClaimAmount: float = Form(...),
    Age: int = Form(...),
    ChronicCount: int = Form(...),
    StayDuration: int = Form(...)
):

    # Convert to DataFrame
    df = pd.DataFrame([{
        "ClaimCount": ClaimCount,
        "AvgClaimAmount": AvgClaimAmount,
        "Age": Age,
        "ChronicCount": ChronicCount,
        "StayDuration": StayDuration
    }])

    # Prediction
    prob = cat_model.predict_proba(df)[0][1]
    anomaly = iso_model.predict(df)[0]

    fraud = 1 if prob > 0.7 or anomaly == -1 else 0

    result = "FRAUD DETECTED 🚨" if fraud == 1 else "NOT FRAUD ✅"

    return f"""
    <html>
        <body style="font-family: Arial; padding: 30px;">
            <h2>Result</h2>
            <h3>{result}</h3>
            <p>Fraud Probability: {prob:.2f}</p>

            <a href="/">Go Back</a>
        </body>
    </html>
    """
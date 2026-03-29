from sklearn.ensemble import IsolationForest
import joblib
import os

def train_isolation_forest(X_train):

    os.makedirs("models", exist_ok=True)

    model = IsolationForest(
        contamination=0.1,
        random_state=42
    )

    model.fit(X_train)

    joblib.dump(model, "models/isolation.pkl")

    return model
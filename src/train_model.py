from catboost import CatBoostClassifier
import joblib
import os

def train_catboost(X_train, y_train):

    os.makedirs("models", exist_ok=True)

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function='Logloss',
        class_weights=[1, 5],
        verbose=100
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "models/catboost.pkl")

    return model
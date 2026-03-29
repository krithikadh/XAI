from catboost import CatBoostClassifier
import joblib

def train_catboost(X_train, y_train):

    model = CatBoostClassifier(verbose=0)

    model.fit(X_train, y_train)

    joblib.dump(model, "models/catboost.pkl")

    return model
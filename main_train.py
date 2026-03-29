from src.preprocessing import load_data, merge_data, clean_data
from src.feature_engineering import create_features
from src.train_model import train_catboost
from src.anomaly_model import train_isolation_forest
from src.preprocessing import prepare_data

# Load
b, i, o, p = load_data()

# Merge
df = merge_data(b, i, o, p)

# Clean
df = clean_data(df)

# Features
df = create_features(df)

# Split
X_train, X_test, y_train, y_test = prepare_data(df)

# Train models
cat_model = train_catboost(X_train, y_train)
iso_model = train_isolation_forest(X_train)
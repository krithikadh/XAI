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

# from src.preprocessing import load_data, merge_data, clean_data, prepare_data
# from src.feature_engineering import create_features
# from src.train_model import train_catboost
# from src.anomaly_model import train_isolation_forest

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# import numpy as np
# import shap
# from lime.lime_tabular import LimeTabularExplainer


# # -------------------------------
# # 1. LOAD & PREPROCESS DATA
# # -------------------------------
# b, i, o, p = load_data()
# df = merge_data(b, i, o, p)
# df = clean_data(df)
# df = create_features(df)

# X_train, X_test, y_train, y_test = prepare_data(df)


# # -------------------------------
# # 2. TRAIN MODELS
# # -------------------------------
# cat_model = train_catboost(X_train, y_train)
# iso_model = train_isolation_forest(X_train)


# # -------------------------------
# # 3. CATBOOST EVALUATION
# # -------------------------------
# print("\n==============================")
# print("📊 CATBOOST MODEL EVALUATION")
# print("==============================")

# y_pred_cat = cat_model.predict(X_test)

# print("Accuracy :", accuracy_score(y_test, y_pred_cat))
# print("Precision:", precision_score(y_test, y_pred_cat))
# print("Recall   :", recall_score(y_test, y_pred_cat))
# print("F1 Score :", f1_score(y_test, y_pred_cat))

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred_cat))

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred_cat))


# # -------------------------------
# # 4. ISOLATION FOREST EVALUATION
# # -------------------------------
# print("\n==============================")
# print("📊 ISOLATION FOREST EVALUATION")
# print("==============================")

# iso_preds = iso_model.predict(X_test)

# # Convert (-1 → fraud, 1 → normal)
# iso_preds = np.where(iso_preds == -1, 1, 0)

# print("Accuracy :", accuracy_score(y_test, iso_preds))
# print("Precision:", precision_score(y_test, iso_preds))
# print("Recall   :", recall_score(y_test, iso_preds))
# print("F1 Score :", f1_score(y_test, iso_preds))

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, iso_preds))


# # -------------------------------
# # 5. ENSEMBLE EVALUATION
# # -------------------------------
# print("\n==============================")
# print("📊 ENSEMBLE MODEL EVALUATION")
# print("==============================")

# cat_probs = cat_model.predict_proba(X_test)[:, 1]
# iso_raw = iso_model.predict(X_test)

# ensemble_preds = []

# for prob, iso in zip(cat_probs, iso_raw):

#     anomaly_flag = 1 if iso == -1 else 0

#     if prob > 0.7 and anomaly_flag == 1:
#         pred = 1
#     elif prob > 0.8:
#         pred = 1
#     elif anomaly_flag == 1 and prob > 0.5:
#         pred = 1
#     else:
#         pred = 0

#     ensemble_preds.append(pred)

# print("Accuracy :", accuracy_score(y_test, ensemble_preds))
# print("Precision:", precision_score(y_test, ensemble_preds))
# print("Recall   :", recall_score(y_test, ensemble_preds))
# print("F1 Score :", f1_score(y_test, ensemble_preds))

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, ensemble_preds))


# # -------------------------------
# # 6. SHAP ANALYSIS (GLOBAL)
# # -------------------------------
# print("\n==============================")
# print("🔍 SHAP FEATURE IMPORTANCE")
# print("==============================")

# explainer = shap.Explainer(cat_model)
# shap_values = explainer(X_test[:100])  # sample for speed

# mean_shap = np.abs(shap_values.values).mean(axis=0)

# feature_importance = list(zip(X_test.columns, mean_shap))
# feature_importance.sort(key=lambda x: x[1], reverse=True)

# print("\nTop 5 Important Features:")
# for f, v in feature_importance[:5]:
#     print(f"{f}: {v:.4f}")


# # -------------------------------
# # 7. LIME ANALYSIS (LOCAL)
# # -------------------------------
# print("\n==============================")
# print("🧠 LIME EXPLANATION (SAMPLE)")
# print("==============================")

# lime_explainer = LimeTabularExplainer(
#     training_data=np.array(X_train),
#     feature_names=X_train.columns.tolist(),
#     mode='classification'
# )

# sample = X_test.iloc[0]

# lime_exp = lime_explainer.explain_instance(
#     sample.values,
#     cat_model.predict_proba
# )

# print("\nLIME Explanation for first test sample:")
# for feature, weight in lime_exp.as_list():
#     print(f"{feature}: {weight:.4f}")


# # -------------------------------
# # 8. FINAL MESSAGE
# # -------------------------------
# print("\n✅ FULL MODEL EVALUATION COMPLETED!")
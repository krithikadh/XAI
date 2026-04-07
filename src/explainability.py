import shap
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

def get_shap_explanation(model, X):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    return shap_values

def get_lime_explanation(model, X_train, instance):
    explainer = LimeTabularExplainer(
        training_data=np.array(X_train),
        mode='classification'
    )
    exp = explainer.explain_instance(
        instance,
        model.predict_proba
    )
    return exp.as_list()
def final_prediction(cat_model, iso_model, X):
    prob = cat_model.predict_proba(X)[:, 1]
    anomaly = iso_model.predict(X)
    anomaly_flag = (anomaly == -1).astype(int) #(-1 = anamoly)
    final = []
    for p, a in zip(prob, anomaly_flag):
        if p > 0.7 or a == 1:
            final.append(1)
        else:
            final.append(0)
    return final, prob
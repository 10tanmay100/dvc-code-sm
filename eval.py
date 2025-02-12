from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score
import joblib
import pandas as pd
from utils import save_json
import json

def evaluate_model(Xtest, ytest,model_path="models/model.pkl"):
    
    model = joblib.load(model_path)
    y_pred=model.predict(Xtest)

    metrics={
        "Precision": precision_score(ytest.values.ravel(),y_pred),
        "Recall": recall_score(ytest.values.ravel(),y_pred),
        "F1": f1_score(ytest.values.ravel(),y_pred),
        "roc_auc_score": roc_auc_score(ytest.values.ravel(),y_pred)
    }

    save_json("metrics.json",metrics)

    


    return {"Precision": precision_score(ytest.values.ravel(),y_pred),"Recall": recall_score(ytest.values.ravel(),y_pred),"F1": f1_score(ytest.values.ravel(),y_pred),"roc_auc_score":roc_auc_score(ytest.values.ravel(),y_pred)}


if __name__ == "__main__":
        X=pd.read_csv(r"splitted_data\xtest.csv")
        y=pd.read_csv(r"splitted_data\ytest.csv")
        print(evaluate_model(X,y))


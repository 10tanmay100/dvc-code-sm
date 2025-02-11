from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score
import joblib
import pandas as pd
import json
from dvclive import Live

def evaluate_model(Xtest, ytest,model_path="models/model.pkl"):
    # Load the trained model
    # with Live() as live:
    model = joblib.load(model_path)
    y_pred=model.predict(Xtest)
        # live.log_metric("Precision",precision_score(ytest.values.ravel(),y_pred))
        # live.log_metric("Recall",recall_score(ytest.values.ravel(),y_pred))
        # live.log_metric("F1",f1_score(ytest.values.ravel(),y_pred))
        # live.log_metric("roc_auc_score",roc_auc_score(ytest.values.ravel(),y_pred))

    metrics={
        "Precision": precision_score(ytest.values.ravel(),y_pred),
        "Recall": recall_score(ytest.values.ravel(),y_pred),
        "F1": f1_score(ytest.values.ravel(),y_pred),
        "roc_auc_score": roc_auc_score(ytest.values.ravel(),y_pred)
    }

    with open("metrics.json","w") as f:
          json.dump(metrics, f, indent=4)


    return {"Precision": precision_score(ytest.values.ravel(),y_pred),"Recall": recall_score(ytest.values.ravel(),y_pred),"F1": f1_score(ytest.values.ravel(),y_pred),"roc_auc_score":roc_auc_score(ytest.values.ravel(),y_pred)}


if __name__ == "__main__":
        X=pd.read_csv(r"splitted_data\xtest.csv")
        y=pd.read_csv(r"splitted_data\ytest.csv")
        print(evaluate_model(X,y))


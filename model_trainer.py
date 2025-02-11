from sklearn.neighbors import KNeighborsClassifier
import joblib
import pandas as pd
from utils import load_params
from dvclive import Live

def train_model(X,y,model_save_path="models/model.pkl"):
    n_neighbors=load_params("params.yaml")["model_trainer"]["n_neighbors"]
    # with Live() as live:
    #     live.log_param("epochs", n_neighbors)

    knn_classifier = KNeighborsClassifier(n_neighbors = n_neighbors)
    knn_classifier.fit(X,y)
    joblib.dump(knn_classifier, model_save_path)
        # live.log_artifact(model_save_path, type="model")
    return True

if __name__ == "__main__":
    # Assuming X and y are your training data
    X=pd.read_csv(r"splitted_data\xtrain.csv")
    y=pd.read_csv(r"splitted_data\ytrain.csv")
    train_model(X, y)
    print("Model saved successfully")
    
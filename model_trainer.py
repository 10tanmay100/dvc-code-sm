from sklearn.neighbors import KNeighborsClassifier
import joblib
import pandas as pd
from utils import load_params

def train_model(X,y,model_save_path="models/model.pkl"):
    n_neighbors_param=load_params("params.yaml")["model_training"]["n_neighbors_param"]
    knn_classifier = KNeighborsClassifier(n_neighbors = n_neighbors_param)
    knn_classifier.fit(X,y)
    joblib.dump(knn_classifier, model_save_path)
    
    return True

if __name__ == "__main__":
    # Assuming X and y are your training data
    X=pd.read_csv(r"splitted_data\xtrain.csv")
    y=pd.read_csv(r"splitted_data\ytrain.csv")
    train_model(X, y)
    print("Model saved successfully")
    
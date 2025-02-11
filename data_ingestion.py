import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from utils import load_params

def load_dataset(data_path):
    df = pd.read_csv(data_path)
    y = df['target']
    X = df.drop(['target'], axis = 1)
    return X,y

def split_dataset(X,y):
    test_size_param=load_params("params.yaml")["data_ingestion"]["test_size"]
    Xtrain, Xtest,Ytrain,Ytest = train_test_split(X,y, test_size=test_size_param, random_state=42)
    os.makedirs("splitted_data",exist_ok=True)
    Xtrain.to_csv("splitted_data/xtrain.csv",index=False), Xtest.to_csv("splitted_data/xtest.csv",index=False),Ytrain.to_csv("splitted_data/ytrain.csv",index=False),Ytest.to_csv("splitted_data/ytest.csv",index=False)


if __name__ == "__main__":
    data_path = 'data/heart.csv'
    X, y = load_dataset(data_path)
    split_dataset(X, y)
    

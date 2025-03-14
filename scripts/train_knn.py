import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib

X_train_data_path = '..\\data\\X_train_data.csv'
y_train_data_path = '..\\data\\y_train_data.csv'
knn_model_save_path = '..\\models\\knn_model.pkl'

def train_knn(X_train, y_train, k=5):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn

if __name__ == '__main__':
    X_train = pd.read_csv(X_train_data_path)
    y_train = pd.read_csv(y_train_data_path).squeeze()

    knn_model = train_knn(X_train, y_train)
    joblib.dump(knn_model, knn_model_save_path)
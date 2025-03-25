from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import json
from sklearn.model_selection import GridSearchCV


X_train_data_path = '..\\data\\X_train_data.csv'
y_train_data_path = '..\\data\\y_train_data.csv'
model_save_path = '..\\models\\knn_model.pkl'
best_params_save_path = '..\\results\\best_params_knn.json'

def load_data():
    X_train = pd.read_csv(X_train_data_path)
    y_train = pd.read_csv(y_train_data_path).squeeze()
    return X_train, y_train

def grid_search(X_train, y_train):
    param_grid = {
        'n_neighbors': [9, 11, 15, 20],
        'weights': ['uniform'],
        'metric': ['euclidean', 'manhattan']
    }
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best hyperparameters: ", grid_search.best_params_)

    # En iyi parametreleri JSON dosyasına kaydet
    with open(best_params_save_path, 'w') as f:
        json.dump(grid_search.best_params_, f, indent=4)
    print(f"Best parameters saved to {best_params_save_path}")

    return grid_search.best_params_

def train_knn(X_train, y_train, best_params):
    knn_model = KNeighborsClassifier(**best_params)
    knn_model.fit(X_train, y_train)
    joblib.dump(knn_model, model_save_path)
    print(f"Model saved to {model_save_path}")

    return knn_model

if __name__ == '__main__':
    # Veriyi yükle
    X_train, y_train = load_data()

    # Grid search ile hiperparametre optimizasyonu
    best_params = grid_search(X_train, y_train)

    # Modeli eğit
    knn_model = train_knn(X_train, y_train, best_params)

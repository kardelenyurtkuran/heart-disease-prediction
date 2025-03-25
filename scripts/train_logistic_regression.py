import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd
import joblib
import json
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

X_train_data_path = '..\\data\\X_train_data.csv'
y_train_data_path = '..\\data\\y_train_data.csv'
model_save_path = '..\\models\\logistic_regression_model.pkl'
best_params_save_path = '..\\results\\best_params_logistic_regression.json'

def load_data():
    X_train = pd.read_csv(X_train_data_path)
    y_train = pd.read_csv(y_train_data_path).squeeze()
    return X_train, y_train

def grid_search(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'class_weight': [None, 'balanced', {0: 1, 1: 1.5}]
    }
    dt = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best hyperparameters: ", grid_search.best_params_)

    # En iyi parametreleri JSON dosyasına kaydet
    with open(best_params_save_path, 'w') as f:
        json.dump(grid_search.best_params_, f, indent=4)
    print(f"Best parameters saved to {best_params_save_path}")

    return grid_search.best_params_

def train_logistic_regression(X_train, y_train, best_params):
    log_reg = LogisticRegression(**best_params, random_state=42)
    log_reg.fit(X_train, y_train)
    joblib.dump(log_reg, model_save_path)
    print(f"Model saved to {model_save_path}")

    return log_reg

if __name__ == '__main__':
    # Veriyi yükle
    X_train, y_train = load_data()

    # Grid search ile hiperparametre optimizasyonu
    best_params = grid_search(X_train, y_train)

    # Modeli eğit
    log_reg_model = train_logistic_regression(X_train, y_train, best_params)

    print(pd.DataFrame({'feature': X_train.columns, 'coefficient': log_reg_model.coef_[0]}).sort_values(
        by='coefficient', ascending=False))
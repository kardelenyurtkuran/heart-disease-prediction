import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
import joblib
import json
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

X_train_data_path = '..\\data\\X_train_data.csv'
y_train_data_path = '..\\data\\y_train_data.csv'
model_save_path = '..\\models\\random_forest_model.pkl'
best_params_save_path = '..\\results\\best_params_random_forest.json'

def load_data():
    X_train = pd.read_csv(X_train_data_path)
    y_train = pd.read_csv(y_train_data_path).squeeze()
    return X_train, y_train

def grid_search(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': [None, 'balanced']
    }
    dt = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best hyperparameters: ", grid_search.best_params_)

    # En iyi parametreleri JSON dosyasına kaydet
    with open(best_params_save_path, 'w') as f:
        json.dump(grid_search.best_params_, f, indent=4)
    print(f"Best parameters saved to {best_params_save_path}")

    return grid_search.best_params_

def train_random_forest(X_train, y_train, best_params):
    """Trains and returns the Random Forest model"""
    rf_model = RandomForestClassifier(**best_params, random_state=42)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, model_save_path)
    print(f"Model saved to {model_save_path}")

    return rf_model


if __name__ == '__main__':
    # Veriyi yükle
    X_train, y_train = load_data()

    # Grid search ile hiperparametre optimizasyonu
    best_params = grid_search(X_train, y_train)

    # Modeli eğit
    rf_model = train_random_forest(X_train, y_train, best_params)

    # Feature importance
    print(pd.DataFrame({'feature': X_train.columns, 'importance': rf_model.feature_importances_}).sort_values(
        by='importance', ascending=False))
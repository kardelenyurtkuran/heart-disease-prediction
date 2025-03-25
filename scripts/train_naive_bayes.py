import pandas as pd
import joblib
import json
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

# File paths
X_train_data_path = '..\\data\\X_train_data.csv'
y_train_data_path = '..\\data\\y_train_data.csv'
model_save_path = '..\\models\\naive_bayes_model.pkl'
best_params_save_path = '..\\results\\best_params_naive_bayes.json'

def load_data():
    X_train = pd.read_csv(X_train_data_path)
    y_train = pd.read_csv(y_train_data_path).squeeze()
    return X_train, y_train

def grid_search(X_train, y_train):
    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
    }
    nb = GaussianNB()
    grid_search = GridSearchCV(nb, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best hyperparameters: ", grid_search.best_params_)

    with open(best_params_save_path, 'w') as f:
        json.dump(grid_search.best_params_, f, indent=4)
    print(f"Best parameters saved to {best_params_save_path}")

    return grid_search.best_params_

def train_naive_bayes(X_train, y_train, best_params):
    nb_model = GaussianNB(**best_params)
    nb_model.fit(X_train, y_train)
    joblib.dump(nb_model, model_save_path)
    print(f"Model saved to {model_save_path}")
    return nb_model

if __name__ == '__main__':
    # Veriyi yükle
    X_train, y_train = load_data()

    # Grid search ile hiperparametre optimizasyonu
    best_params = grid_search(X_train, y_train)

    # Modeli eğit
    nb_model = train_naive_bayes(X_train, y_train, best_params)

    # Permutation importance ile özellik önemini hesapla
    perm_importance = permutation_importance(nb_model, X_train, y_train, n_repeats=10, random_state=42, scoring='f1')
    print(pd.DataFrame({'feature': X_train.columns, 'importance': perm_importance.importances_mean}).sort_values(
        by='importance', ascending=False))
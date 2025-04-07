import pandas as pd
import joblib
import json
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

# File paths
X_train_data_path = '..\\data\\X_train_data.csv'
y_train_data_path = '..\\data\\y_train_data.csv'
model_save_path = '..\\models\\svm_model.pkl'
best_params_save_path = '..\\results\\best_params_svm.json'

def load_data():
    X_train = pd.read_csv(X_train_data_path)
    y_train = pd.read_csv(y_train_data_path).squeeze()
    return X_train, y_train

def grid_search(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'class_weight': [None, 'balanced', {0: 1, 1: 1.5}]
    }
    svm = SVC(probability=True, random_state=42)  # probability=True, eşik ayarı için gerekli
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best hyperparameters: ", grid_search.best_params_)

    with open(best_params_save_path, 'w') as f:
        json.dump(grid_search.best_params_, f, indent=4)
    print(f"Best parameters saved to {best_params_save_path}")

    return grid_search.best_params_

def train_svm(X_train, y_train, best_params):
    svm_model = SVC(**best_params, probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    joblib.dump(svm_model, model_save_path)
    print(f"Model saved to {model_save_path}")
    return svm_model

if __name__ == '__main__':
    # Veriyi yükle
    X_train, y_train = load_data()

    # Grid search ile hiperparametre optimizasyonu
    best_params = grid_search(X_train, y_train)

    # Modeli eğit
    svm_model = train_svm(X_train, y_train, best_params)

    # Permutation importance ile özellik önemini hesapla
    perm_importance = permutation_importance(svm_model, X_train, y_train, n_repeats=10, random_state=42, scoring='f1')
    print(pd.DataFrame({'feature': X_train.columns, 'importance': perm_importance.importances_mean}).sort_values(
        by='importance', ascending=False))
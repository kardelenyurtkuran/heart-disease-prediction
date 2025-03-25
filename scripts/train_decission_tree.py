import pandas as pd
import joblib
import json
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# File paths
X_train_data_path = '..\\data\\X_train_data.csv'
y_train_data_path = '..\\data\\y_train_data.csv'
model_save_path = '..\\models\\decision_tree_model.pkl'
best_params_save_path = '..\\results\\best_params_decision_tree.json'

def load_data():
    X_train = pd.read_csv(X_train_data_path)
    y_train = pd.read_csv(y_train_data_path).squeeze()
    return X_train, y_train

def grid_search(X_train, y_train):
    param_grid = {
        'criterion': ["gini", "entropy"],
        'max_depth': [3, 5, 10, 15, 20],
        'min_samples_leaf': [5, 10, 20, 30, 40],
        'min_samples_split': [5, 10, 20, 30, 40],
        'max_features': [None, 'sqrt', 'log2'],
        'ccp_alpha': [0.0, 0.0001, 0.001, 0.05]
    }
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best hyperparameters: ", grid_search.best_params_)

    # En iyi parametreleri JSON dosyasına kaydet
    with open(best_params_save_path, 'w') as f:
        json.dump(grid_search.best_params_, f, indent=4)
    print(f"Best parameters saved to {best_params_save_path}")

    return grid_search.best_params_

def train_decision_tree(X_train, y_train, best_params):
    dt_model = DecisionTreeClassifier(random_state=42, **best_params)
    dt_model.fit(X_train, y_train)

    # Modeli kaydet
    joblib.dump(dt_model, model_save_path)
    print(f"Model saved to {model_save_path}")

    return dt_model

def tree_visualization(dt_model, X_train):
    plt.figure(figsize=(20, 30))
    plot_tree(dt_model,
              feature_names=X_train.columns,
              class_names=['0', '1'],
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title("Decision Tree Visualization")
    plt.savefig("..\\results\\decision_tree.png")
    plt.show()

if __name__ == "__main__":
    # Veriyi yükle
    X_train, y_train = load_data()

    # Grid search ile hiperparametre optimizasyonu
    best_params = grid_search(X_train, y_train)

    # Modeli eğit
    dt_model = train_decision_tree(X_train, y_train, best_params)

    # Feature importance
    print(pd.DataFrame({'feature': X_train.columns, 'importance': dt_model.feature_importances_}).sort_values(
        by='importance', ascending=False))

    # Ağacı görselleştir
    tree_visualization(dt_model, X_train)
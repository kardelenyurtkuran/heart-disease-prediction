import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

X_train_data_path = '..\\data\\X_train_data.csv'
y_train_data_path = '..\\data\\y_train_data.csv'
model_save_path = '..\\models\\decission_tree_model.pkl'

def train_decission_tree(X_train, y_train):
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    return dt_model

if __name__ == '__main__':
    X_train = pd.read_csv(X_train_data_path)
    y_train = pd.read_csv(y_train_data_path).squeeze()

    dt_model = train_decission_tree(X_train, y_train)
    joblib.dump(dt_model, model_save_path)
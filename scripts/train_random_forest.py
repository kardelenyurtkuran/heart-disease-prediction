import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

X_train_data_path = '..\\data\\X_train_data.csv'
y_train_data_path = '..\\data\\y_train_data.csv'
model_save_path = '..\\models\\random_forest_model.pkl'

def train_random_forest(X_train, y_train, n_estimators):
    """Trains and returns the Random Forest model"""
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

if __name__ == '__main__':
    #Load dataset
    X_train = pd.read_csv(X_train_data_path)
    y_train = pd.read_csv(y_train_data_path).squeeze()

    #Train Model
    rf_model = train_random_forest(X_train, y_train, 100)

    #Save Model
    joblib.dump(rf_model, model_save_path)
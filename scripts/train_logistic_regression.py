import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

X_train_data_path = '..\\data\\X_train_data.csv'
y_train_data_path = '..\\data\\y_train_data.csv'
model_save_path = '..\\models\\logistic_regression_model.pkl'

def train_logistic_regression(X_train, y_train):
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    return log_reg

if __name__ == '__main__':
    X_train = pd.read_csv(X_train_data_path)
    y_train = pd.read_csv(y_train_data_path).squeeze()
    log_reg_model = train_logistic_regression(X_train, y_train)
    joblib.dump(log_reg_model, model_save_path)
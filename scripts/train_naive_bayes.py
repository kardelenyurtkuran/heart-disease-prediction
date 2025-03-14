import pandas as pd
from sklearn.naive_bayes import GaussianNB
import joblib

X_train_data_path = '..\\data\\X_train_data.csv'
y_train_data_path = '..\\data\\y_train_data.csv'
model_save_path = '..\\models\\naive_bayes_model.pkl'

def train_naive_bayes(X_train, y_train):
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    return nb_model

if __name__ == '__main__':
    X_train = pd.read_csv(X_train_data_path)
    y_train = pd.read_csv(y_train_data_path).squeeze()

    nb_model = train_naive_bayes(X_train, y_train)
    joblib.dump(nb_model, model_save_path)
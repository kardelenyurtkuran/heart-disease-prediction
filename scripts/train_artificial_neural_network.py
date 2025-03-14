import pandas as pd
from sklearn.neural_network import MLPClassifier
import joblib

X_train_data_path = '..\\data\\X_train_data.csv'
y_train_data_path = '..\\data\\y_train_data.csv'
model_save_path = '..\\models\\ann_model.pkl'

def train_ann(X_train, y_train):
    """Trains and returns the Artificial Neural Networks (ANN) model."""
    ann_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500, random_state=42)
    ann_model.fit(X_train, y_train)
    return ann_model

if __name__ == '__main__':
    #Load dataset
    X_train = pd.read_csv(X_train_data_path)
    y_train = pd.read_csv(y_train_data_path)

    #Train model
    ann_model = train_ann(X_train, y_train)

    #Save model
    joblib.dump(ann_model, model_save_path)
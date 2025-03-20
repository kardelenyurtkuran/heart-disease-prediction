import joblib
import pandas as pd

X_train_data_path = '..\\data\\X_train_data.csv'
model_save_path = '..\\models\\decision_tree_model.pkl'


def feature_importance(X_train, dt_model):
    feature_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': dt_model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    return feature_importances


if __name__ == '__main__':
    X_train = pd.read_csv(X_train_data_path)
    dt_model = joblib.load(model_save_path)
    print(feature_importance(X_train, dt_model))
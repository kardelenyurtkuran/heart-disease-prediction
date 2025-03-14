import pandas as pd
from pkg_resources import split_sections
from scipy.signal import dfreqresp
from sklearn.model_selection import StratifiedKFold
from sqlalchemy.dialects.mssql.information_schema import columns

cleaned_data_path = '..\\data\\cleaned_heart_statlog.csv'
X_train_data_path = '..\\data\\X_train_data.csv'
X_test_data_path = '..\\data\\X_test_data.csv'
y_train_data_path = '..\\data\\y_train_data.csv'
y_test_data_path = '..\\data\\y_test_data.csv'

def cross_validation_split(df, target_column="target", n_splits=5, random_state=42):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []

    for train_index, test_index in skf.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        splits.append([X_train, X_test, y_train, y_test])

    return splits

if __name__ == "__main__":
    df = pd.read_csv(cleaned_data_path)
    splits = cross_validation_split(df)
    X_train, X_test, y_train, y_test = splits[0]
    X_train.to_csv(X_train_data_path, index=False)
    X_test.to_csv(X_test_data_path, index=False)
    y_train.to_csv(y_train_data_path, index=False)
    y_test.to_csv(y_test_data_path, index=False)
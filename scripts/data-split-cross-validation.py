import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sqlalchemy.dialects.mssql.information_schema import columns


def cross_validation_split(df, target_column="target", n_splits=5, random_state=42):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []

    for train_index, test_index in skf.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        splits.append([X_train, X_test, y_train, y_test])
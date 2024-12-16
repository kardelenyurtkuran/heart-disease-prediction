import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, mean_squared_error


class BaggedSVM:
    def __init__(self, n_estimators=10, feature_fraction=0.8, model_type="classification"):
        self.n_estimators = n_estimators
        self.feature_fraction = feature_fraction
        self.model_type = model_type
        self.models = []
        self.selected_features = []

    def fit(self, X, y):
        n_features = X.shape[1]
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            X_sample, y_sample = resample(X, y)

            # Randomly select features
            feature_indices = np.random.choice(range(n_features),
                                               size=int(n_features * self.feature_fraction),
                                               replace=False)
            X_sample_selected = X_sample[:, feature_indices]

            # Train an SVM/SVR
            if self.model_type == "classification":
                model = SVC(probability=True)
            elif self.model_type == "regression":
                model = SVR()
            else:
                raise ValueError("Invalid model_type. Choose 'classification' or 'regression'.")

            model.fit(X_sample_selected, y_sample)

            # Store the model and selected features
            self.models.append(model)
            self.selected_features.append(feature_indices)

    def predict(self, X):
        predictions = []
        for model, feature_indices in zip(self.models, self.selected_features):
            X_selected = X[:, feature_indices]
            if self.model_type == "classification":
                predictions.append(model.predict_proba(X_selected)[:, 1])  # For classification
            elif self.model_type == "regression":
                predictions.append(model.predict(X_selected))  # For regression

        # Aggregate predictions
        if self.model_type == "classification":
            final_prediction = (np.mean(predictions, axis=0) > 0.5).astype(int)
        elif self.model_type == "regression":
            final_prediction = np.mean(predictions, axis=0)
        return final_prediction


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    # Veri setini yükle
    data = pd.read_csv('cleaned_scaled_heart_disease_data.csv')
    X = data.drop("target", axis=1).values  # Hedef sütunu çıkar
    y = data["target"].values  # Hedef sütununu al

    # Eğitim ve test setine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modeli oluştur ve eğit
    bagged_svm = BaggedSVM(n_estimators=10, feature_fraction=0.8, model_type="classification")
    bagged_svm.fit(X_train, y_train)

    # Tahmin yap ve değerlendirme
    y_pred = bagged_svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Set Accuracy:", accuracy)
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# File paths
X_train_path = '..\\data\\X_train_data.csv'
y_train_path = '..\\data\\y_train_data.csv'
X_test_path = '..\\data\\X_test_data.csv'
y_test_path = '..\\data\\y_test_data.csv'

model_paths = {
    'KNN': '..\\models\\knn_model.pkl',
    'Naive Bayes': '..\\models\\naive_bayes_model.pkl',
    'Logistic Regression': '..\\models\\logistic_regression_model.pkl',
    'Decision Tree': '..\\models\\decision_tree_model.pkl',
    'Random Forest': '..\\models\\random_forest_model.pkl',
    'ANN': '..\\models\\ann_model.pkl',
    'SVM': '..\\models\\svm_model.pkl'
}

# Load Data
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path).squeeze()
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path).squeeze()

results = []

# Measure performance for each model
for model_name, model_path in model_paths.items():
    model = joblib.load(model_path)  # Load model

    # Get Train and Test predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate Performance Metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    test_precision = precision_score(y_test, y_test_pred, average='weighted')

    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')

    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    # Add to results
    results.append([model_name, train_accuracy, test_accuracy, train_precision,
                    test_precision, train_recall, test_recall, train_f1, test_f1])

    # Confusion Matrix for Test Set
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name} (Test Set)")
    plt.savefig(f"..\\results\\confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.close()

# Save results to CSV
columns = ["Model", "Train Accuracy", "Test Accuracy", "Train Precision", "Test Precision",
           "Train Recall", "Test Recall", "Train F1 Score", "Test F1 Score"]
results_df = pd.DataFrame(results, columns=columns)
results_df.to_csv("..\\results\\model_performance.csv", index=False)

print(results_df)
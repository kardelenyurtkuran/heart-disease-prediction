import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Veri setini yükle
df = pd.read_csv('cleaned_scaled_heart_disease_data.csv')

X = df.drop(columns=['target'])
y = df['target']

# Eğitim ve test setini oluştur
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Random Forest modelini oluştur ve eğit (underfitting için yapılandırma)
rf_model = RandomForestClassifier(
    random_state=42,
    n_estimators=50,     # Az ağaç
    max_depth=10,         # Daha sınırlı derinlik
    min_samples_split=10, # Daha fazla veri ile bölsün
    min_samples_leaf=10   # Daha fazla veri ile yaprak düğüm
)
rf_model.fit(X_train, y_train)

# Tahminler
y_pred_rf = rf_model.predict(X_test)

# Eğitim doğruluğu
train_accuracy = rf_model.score(X_train, y_train)
print(f"Train Accuracy: {train_accuracy}")

# Test doğruluğu
accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Test Accuracy: {accuracy}")

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf)
print("\nConfusion Matrix:\n", conf_matrix)

# ROC Curve ve AUC
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]  # Pozitif sınıf için olasılık
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_rf)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='Random Forest (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

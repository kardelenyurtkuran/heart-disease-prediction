import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Veri setini yükle
df = pd.read_csv('cleaned_scaled_heart_disease_data.csv')

X = df.drop(columns=['target'])
y = df['target']

# Eğitim ve test setini oluştur (eğitim seti küçültülüyor)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.85, random_state=42)

# Random Forest modelini oluştur ve eğit (karmaşık model)
rf_model = RandomForestClassifier(
    random_state=42,
    n_estimators=3000,  # Çok sayıda ağaç
    max_depth=None,     # Sınırsız derinlik
    min_samples_split=2,  # Split için minimum veri sayısı
    min_samples_leaf=1,   # Yaprak düğümlerde minimum veri sayısı
    max_features=None     # Tüm öznitelikler kullanılacak
)
rf_model.fit(X_train, y_train)

# Eğitim doğruluğu
train_accuracy = rf_model.score(X_train, y_train)
print(f"Train Accuracy: {train_accuracy}")

# Test doğruluğu
y_pred_rf = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Test Accuracy: {test_accuracy}")

# Classification Report ve Confusion Matrix
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

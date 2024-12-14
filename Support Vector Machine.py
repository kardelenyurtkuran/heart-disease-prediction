import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükle
df = pd.read_csv('cleaned_scaled_heart_disease_data.csv')

# Bağımsız ve bağımlı değişkenleri ayır
X = df.drop(columns=['target'])  # Bağımlı değişken
y = df['target']

# Eğitim ve test setlerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SVM modelini tanımla ve eğit
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Tahmin yap
y_pred = svm_model.predict(X_test)

# Model performansını değerlendirme
print("Doğruluk Oranı:", accuracy_score(y_test, y_pred))
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print("\nKarmaşıklık Matrisi:\n", confusion_matrix(y_test, y_pred))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Veri setini yükle
df = pd.read_csv('cleaned_scaled_heart_disease_data.csv')

# Bağımlı ve bağımsız değişkenleri ayır
X = df.drop(columns=['target'])
y = df['target']

# Eğitim ve test veri bölümü
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Yapay Sinir Ağı modelini oluştur
nn_model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42, activation='relu', solver='adam')
nn_model.fit(X_train, y_train)

# Model tahmini ve performans değerlendirmesi
y_pred = nn_model.predict(X_test)

# Doğruluk Oranı
print("Doğruluk Oranı:", accuracy_score(y_test, y_pred))

# Sınıflandırma Raporu
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))

# Karmaşıklık Matrisi
print("\nKarmaşıklık Matrisi:\n", confusion_matrix(y_test, y_pred))

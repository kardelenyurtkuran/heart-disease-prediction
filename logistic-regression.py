import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('cleaned_scaled_heart_disease_data.csv')

# Hedef ve özellikleri ayır
X = df.drop('target', axis=1)  # Hedef değişken 'target' varsayılır.
y = df['target']  # Hedef değişken (kalp hastalığı var/yok)

# Veri setini eğitim ve test setine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model =LogisticRegression(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Doğruluk, hata matrisi ve sınıflama raporu
print("Doğruluk Oranı:", accuracy_score(y_test, y_pred))
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print("\nKarmaşıklık Matrisi:\n", confusion_matrix(y_test, y_pred))

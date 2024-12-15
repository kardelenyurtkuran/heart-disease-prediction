# Gerekli kütüphaneleri yükle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Veri setini yükle
df = pd.read_csv('cleaned_scaled_heart_disease_data.csv')

# Bağımlı ve bağımsız değişkenleri ayır
X = df.drop('target', axis=1)  # Hedef değişkeni çıkar
y = df['target']               # Hedef değişken

# Veri setini eğitim ve test olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar ağacı modelini oluştur ve eğit
clf = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# Test seti üzerinde tahmin yap
y_pred = clf.predict(X_test)

# Performans metriklerini yazdır
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk Oranı: {accuracy:.2f}")

print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print("\nKarmaşıklık Matrisi:\n", confusion_matrix(y_test, y_pred))

# Karar ağacını görselleştir
plt.figure(figsize=(20, 12))
plot_tree(clf, feature_names=X.columns, class_names=['No Disease', 'Disease'], filled=True, fontsize=7)
plt.title("Karar Ağacı Görselleştirmesi")
plt.show()

# Özellik önem sıralamasını görselleştir
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': clf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nÖzellik Önem Sıralaması:")
print(feature_importance)

# Grafikle göster
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel('Özellik Önemi')
plt.title('Karar Ağacı - Özellik Önem Sıralaması')
plt.show()

# Gerekli kütüphaneleri yükle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix

# Veri setini yükle
# Örnek: 'heart.csv' dosyasını kullanıyorsan
df = pd.read_csv('cleaned_scaled_heart_disease_data.csv')

# Bağımlı ve bağımsız değişkenleri ayır
X = df.drop('target', axis=1)  # 'target' hedef değişken
y = df['target']

# Veri setini eğitim ve test olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar ağacı modelini oluştur ve eğit
clf = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# Modeli değerlendirme
y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Karar ağacını görselleştir
import matplotlib.pyplot as plt

# Karar ağacını daha büyük bir boyutta görselleştirme
plt.figure(figsize=(20, 12))  # Boyutları artır
plot_tree(clf, feature_names=X.columns, class_names=['No Disease', 'Disease'], filled=True, fontsize=10)
plt.show()


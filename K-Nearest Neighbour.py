import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV



df = pd.read_csv('cleaned_scaled_heart_disease_data.csv')
print(df.info())
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

#k=3
k=10
#k=500
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# param_grid = {'n_neighbors': list(range(1, 21))}
# grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
# grid.fit(X_train, y_train)
# best_k = grid.best_params_['n_neighbors']
# print("Best k:", best_k)

print("Doğruluk Oranı:", accuracy_score(y_test, y_pred))
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print("\nKarmaşıklık Matrisi:\n", confusion_matrix(y_test, y_pred))



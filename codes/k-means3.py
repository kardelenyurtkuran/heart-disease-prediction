import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Temizlenmiş veri setini yükleyin
data = pd.read_csv('processed_dataset.csv')

# Kümeleme sonuçlarını analiz etmek için listeler
sse = []  # Hatanın Karesinin Toplamı (SSE)
silhouette_scores = []  # Silhouette skorları

# Küme sayısı aralığı (2'den 10'a kadar deneyelim)
k_values = range(2, 11)

# Farklı küme sayıları için K-ortalama uygulayın
for k in k_values:
    kmeans = KMeans(n_clusters=k, init='random', n_init=10, random_state=42)
    kmeans.fit(data)

    # Hatanın Karesinin Toplamı
    sse.append(kmeans.inertia_)

    # Silhouette skoru
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(data, labels))

# Grafikleri çizelim
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(k_values, sse, marker='o')
plt.title("Hatanın Karesinin Toplamı (SSE)")
plt.xlabel("Küme Sayısı")
plt.ylabel("SSE")

plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title("Silhouette Skoru")
plt.xlabel("Küme Sayısı")
plt.ylabel("Silhouette")

plt.tight_layout()
plt.show()

# Optimum küme sayısını belirleme
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Optimum Küme Sayısı (Silhouette ile): {optimal_k}")
print(f"En Yüksek Silhouette Skoru: {max(silhouette_scores)}")
print(f"SSE : {sse}")

# Optimum küme sayısı ile K-ortalama
kmeans_final = KMeans(n_clusters=optimal_k, init='random', n_init=10, random_state=42)
kmeans_final.fit(data)


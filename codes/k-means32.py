import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Temizlenmiş veri setini yükleyin
data = pd.read_csv('processed_dataset.csv')

# Hiyerarşik Kümeleme: Grup Ortalaması ile Linkage Matriksi Oluşturma
linkage_matrix = linkage(data, method='ward', metric='euclidean')

# K-Means için grupları oluştur
max_clusters = 10
cluster_assignments = fcluster(linkage_matrix, max_clusters, criterion='maxclust')

# Silhouette skoru ve SSE hesaplamak için liste
sse = []  # Hatanın Karesinin Toplamı (SSE)
silhouette_scores = []  # Silhouette skoru

for k in range(2, max_clusters + 1):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(data)

    sse.append(kmeans.inertia_)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(data, labels))

# Grafikleri çizelim
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(range(2, max_clusters + 1), sse, marker='o')
plt.title("Hatanın Karesinin Toplamı (SSE)")
plt.xlabel("Küme Sayısı")
plt.ylabel("SSE")

plt.subplot(1, 2, 2)
plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
plt.title("Silhouette Skoru")
plt.xlabel("Küme Sayısı")
plt.ylabel("Silhouette")

plt.tight_layout()
plt.show()

# Optimum küme sayısını belirleme
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Optimum Küme Sayısı (Silhouette ile): {optimal_k}")
print(f"En Yüksek Silhouette Skoru: {max(silhouette_scores)}")
print(f"SSE: {sse}")

# K-Means ile grup ortalamaları kullanılarak kümeleme
kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
kmeans_final.fit(data)

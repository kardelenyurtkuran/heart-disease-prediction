import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


# İşlenmiş veri setini yükleyin
data = pd.read_csv('processed_dataset.csv')

# 1. Hiyerarşik kümeleme (linkage matrisi oluşturma)
# 'ward', 'single', 'complete', 'average' yöntemlerinden birini seçebilirsiniz
linkage_matrix = linkage(data, method='complete', metric='euclidean')

# 2. Dendrogram çizimi
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.title("Dendrogram")
plt.xlabel("Veri Noktaları")
plt.ylabel("Öklid Mesafesi")
plt.show()

# 3. Küme sayısını belirleme
# Örnek olarak 2-10 arasında küme sayısını değerlendirelim
sse = []
silhouette_scores = []

for k in range(2, 11):
    clusters = fcluster(linkage_matrix, k, criterion='maxclust')
    sse.append(np.sum((data.values - np.mean(data.values, axis=0)) ** 2))
    silhouette_scores.append(silhouette_score(data, clusters))

# SSE ve Silhouette skorlarını görselleştirme
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(range(2, 11), sse, marker='o')
plt.title("Hatanın Karesinin Toplamı (SSE)")
plt.xlabel("Küme Sayısı")
plt.ylabel("SSE")

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title("Silhouette Skoru")
plt.xlabel("Küme Sayısı")
plt.ylabel("Silhouette")

plt.tight_layout()
plt.show()

# 4. Optimum küme sayısına karar verin ve etiketleri ekleyin
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Optimum Küme Sayısı (Silhouette ile): {optimal_k}")
print(max(silhouette_scores))
print(f"SSE : {sse}")
# Son olarak küme etiketlerini veri setine ekleme
# final_clusters = fcluster(linkage_matrix, optimal_k, criterion='maxclust')
# data['cluster'] = final_clusters
#
# # Yeni veri setini kaydetme
# data.to_csv('hierarchical_clusters.csv', index=False)
# print("Kümeleme tamamlandı, sonuçlar 'hierarchical_clusters.csv' olarak kaydedildi.")

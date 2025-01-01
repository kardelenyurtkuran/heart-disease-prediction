from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Veri yükleme
df = pd.read_csv("cleaned_scaled_heart_disease_data.csv")
df_numeric = df.select_dtypes(include=[np.number])

# Uzaklık matrisinin hesaplanması
pairwise_dists = pdist(df_numeric, metric='euclidean')

# WCSS ve Silhouette Scores hesaplamaları
wcss = []
silhouette_scores = []
n_clusters_range = range(2, 10)  # 2'den 10'a kadar olan kümelerin sayısı

for n_clusters in n_clusters_range:
    hc = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    cluster_labels = hc.fit_predict(df_numeric)

    # Silhouette Score hesaplanması
    silhouette_avg = silhouette_score(df_numeric, cluster_labels)
    silhouette_scores.append(silhouette_avg)

    # WCSS hesaplanması (her bir kümeye ait mesafeleri toplamak)
    wcss_sum = 0
    for i in range(len(df_numeric)):
        for j in range(i + 1, len(df_numeric)):
            if cluster_labels[i] == cluster_labels[j]:
                wcss_sum += np.linalg.norm(df_numeric.iloc[i] - df_numeric.iloc[j])
    wcss.append(wcss_sum)

# En iyi küme sayısını seçmek için grafik ve karşılaştırma
plt.figure(figsize=(12, 6))

# WCSS Elbow yöntemi
plt.subplot(1, 2, 1)
plt.plot(n_clusters_range, wcss, marker='o')
plt.title('WCSS vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')

# Silhouette Score grafiği
plt.subplot(1, 2, 2)
plt.plot(n_clusters_range, silhouette_scores, marker='o')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

# Optimal küme sayısını seçiyoruz
best_n_clusters_wcss = n_clusters_range[np.argmin(wcss)]
best_n_clusters_silhouette = n_clusters_range[np.argmax(silhouette_scores)]

print(f"Optimal number of clusters based on WCSS: {best_n_clusters_wcss}")
print(f"Optimal number of clusters based on Silhouette Score: {best_n_clusters_silhouette}")

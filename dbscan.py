import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Veri yükleme
df = pd.read_csv("cleaned_scaled_heart_disease_data.csv")
df_numeric = df.select_dtypes(include=[np.number])


# EPS ve Min_samples kombinasyonları
eps_values = [0.01, 0.05, 0.1, 0.2, 0.3]
min_samples_values = [3, 5, 8, 10, 15]

results = []

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(df)

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        results.append((eps, min_samples, n_clusters, cluster_labels))

        print(f"EPS: {eps}, Min_samples: {min_samples}, Number of clusters: {n_clusters}")

# Grafikleri oluşturma
for eps, min_samples, n_clusters, cluster_labels in results:
    plt.figure(figsize=(8, 6))
    plt.scatter(df[:, 0], df[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.6, edgecolors='k')
    plt.title(f'DBSCAN Clustering (EPS: {eps}, Min_samples: {min_samples}, Clusters: {n_clusters})')
    plt.xlabel('Feature 1 (Scaled)')
    plt.ylabel('Feature 2 (Scaled)')
    plt.show()

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Veri setini yükleyin
data = pd.read_csv('processed_dataset.csv')

# Orijinal veriyle KMeans kümeleme
kmeans_original = KMeans(n_clusters=5, random_state=42)
labels_original = kmeans_original.fit_predict(data)
silhouette_original = silhouette_score(data, labels_original)

# Boyut azaltma - PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

# PCA ile KMeans
kmeans_pca = KMeans(n_clusters=5, random_state=42)
labels_pca = kmeans_pca.fit_predict(data_pca)
silhouette_pca = silhouette_score(data_pca, labels_pca)

# Boyut azaltma - t-SNE
tsne = TSNE(n_components=2, random_state=42)
data_tsne = tsne.fit_transform(data)

# t-SNE ile KMeans
kmeans_tsne = KMeans(n_clusters=5, random_state=42)
labels_tsne = kmeans_tsne.fit_predict(data_tsne)
silhouette_tsne = silhouette_score(data_tsne, labels_tsne)

# Sonuçları karşılaştırma
print("Silhouette Skorları:")
print(f"Orijinal Veri ile KMeans: {silhouette_original:.4f}")
print(f"PCA ile KMeans: {silhouette_pca:.4f}")
print(f"t-SNE ile KMeans: {silhouette_tsne:.4f}")

# Grafikle karşılaştırma
plt.figure(figsize=(18, 6))

# Orijinal veri
plt.subplot(3, 1, 1)
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels_original, cmap='viridis', s=50)
plt.title("Orijinal Veri ile KMeans")

# PCA
plt.subplot(3, 1, 2)
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels_pca, cmap='viridis', s=50)
plt.title("PCA ile KMeans")

# t-SNE
plt.subplot(3, 1, 3)
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels_tsne, cmap='viridis', s=50)
plt.title("t-SNE ile KMeans")

plt.tight_layout()
plt.show()

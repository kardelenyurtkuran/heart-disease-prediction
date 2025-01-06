import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv('clustered_heart_disease_data.csv')
df.head()

# Apply PCA
pca = PCA(n_components=2)  # Reducing to 2 dimensions for visualization
pca_result = pca.fit_transform(df.drop('target', axis=1))  # Assuming 'target' is the label column
pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
pca_df['target'] = df['target']

# Explained variance ratio
print(f'Explained variance ratio (PCA): {pca.explained_variance_ratio_}')

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(df.drop('target', axis=1))
tsne_df = pd.DataFrame(tsne_result, columns=['tSNE1', 'tSNE2'])
tsne_df['target'] = df['target']

# K-means clustering
kmeans_pca = KMeans(n_clusters=3, random_state=42)  # Assuming 3 clusters as a starting point
kmeans_pca.fit(pca_df[['PCA1', 'PCA2']])
pca_df['cluster'] = kmeans_pca.labels_

kmeans_tsne = KMeans(n_clusters=3, random_state=42)
kmeans_tsne.fit(tsne_df[['tSNE1', 'tSNE2']])
tsne_df['cluster'] = kmeans_tsne.labels_

# Visualization for PCA results
plt.figure(figsize=(12, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='target', style='cluster', data=pca_df, palette='tab10', s=100)
plt.title('PCA & K-means Clustering')
plt.show()

# Visualization for t-SNE results
plt.figure(figsize=(12, 6))
sns.scatterplot(x='tSNE1', y='tSNE2', hue='target', style='cluster', data=tsne_df, palette='tab10', s=100)
plt.title('t-SNE & K-means Clustering')
plt.show()


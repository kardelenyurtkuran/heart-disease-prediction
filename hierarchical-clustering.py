import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Veri yükleme
df = pd.read_csv("cleaned_scaled_heart_disease_data.csv")
df_numeric = df.select_dtypes(include=[np.number])

# Hiyerarşik kümeleme - Bağlantı matrisinin hesaplanması
linkage_matrix = linkage(df_numeric, method='ward')

# Dendrogram oluşturma
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix, labels=df_numeric.index, leaf_rotation=90, leaf_font_size=8, color_threshold=0.5)
plt.title('Dendrogram')
plt.xlabel('Observations')
plt.ylabel('Distance')
plt.show()

# Optimal küme sayısını belirleme
# Cut dendrogram at a distance threshold to define the number of clusters
# Bu örnek seçilecek kümeleri belirtmek içindir
optimal_clusters = 9  # Bu sayıyı dendrogram'dan seçebilirsiniz
cluster_labels = fcluster(linkage_matrix, t=optimal_clusters, criterion='maxclust')

# K-ortalamalar uygulanarak kümeleri tahmin etme
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['Cluster_Labels'] = kmeans.fit_predict(df_numeric)

# Sonuçları inceleyin
print("Optimal number of clusters:", optimal_clusters)
print("Cluster labels added to the dataframe.")

# Kümelere göre veri inceleme
print(df.groupby('Cluster_Labels').mean())

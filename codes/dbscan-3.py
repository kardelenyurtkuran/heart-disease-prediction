import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Temizlenmiş veri setini yükleyin
data = pd.read_csv('processed_dataset.csv')

# DBSCAN parametrelerinin etkisini analiz etmek için grid arama
eps_values = np.arange(0.1, 2.0, 0.1)  # Epsilon değerleri
min_samples_values = range(3, 11)  # Minimum örnek değerleri

results = []

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data)

        # Gürültü (etiket -1) içeren kümeleme sonuçlarını analiz et
        if len(set(labels)) > 1:  # En az 1 küme varsa değerlendirme yap
            silhouette_avg = silhouette_score(data, labels, metric='euclidean')
            results.append((eps, min_samples, silhouette_avg))

# Sonuçları analiz et ve en iyi parametreleri seç
best_params = max(results, key=lambda x: x[2])  # Silhouette skoruna göre en iyi parametreler
best_eps, best_min_samples, best_silhouette = best_params

print(f"En İyi Parametreler:")
print(f"  Epsilon (eps): {best_eps}")
print(f"  Minimum Örnek (min_samples): {best_min_samples}")
print(f"  Silhouette Skoru: {best_silhouette}")

# En iyi parametrelerle DBSCAN uygulayın
dbscan_best = DBSCAN(eps=best_eps, min_samples=best_min_samples)
labels_best = dbscan_best.fit_predict(data)

# Küme ve gürültü dağılımı
unique_labels, counts = np.unique(labels_best, return_counts=True)
cluster_distribution = dict(zip(unique_labels, counts))

print("Küme ve Gürültü Dağılımı:")
for label, count in cluster_distribution.items():
    cluster_type = "Gürültü" if label == -1 else f"Küme {label}"
    print(f"{cluster_type}: {count} örnek")

# Silhouette skorları için sonuçların görselleştirilmesi
eps_values_plot, min_samples_plot, silhouette_scores_plot = zip(*results)
plt.figure(figsize=(10, 6))
plt.scatter(eps_values_plot, min_samples_plot, c=silhouette_scores_plot, cmap='viridis', s=100)
plt.colorbar(label='Silhouette Skoru')
plt.title("DBSCAN Parametre Analizi")
plt.xlabel("Epsilon (eps)")
plt.ylabel("Minimum Örnek (min_samples)")
plt.show()

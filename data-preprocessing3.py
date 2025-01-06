import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

# CSV dosyasını yükleme
file_path = "heart_statlog_cleveland_hungary_final.csv"  # Dosya yolunu buraya girin
df = pd.read_csv(file_path)

# 1. Eksik veri kontrolü
def check_missing_values(df):
    missing_values = df.isnull().sum()
    print("Eksik değerler:\n", missing_values)
    return missing_values

# 2. Tekrarlayan satırların kaldırılması
def remove_duplicates(df):
    return df.drop_duplicates()

# 3. Sürekli ve kategorik değişkenlerin ayrılması
categorical_features = ["sex", "chest pain type", "fasting blood sugar", "resting ecg", "exercise angina", "ST slope", "target"]
continuous_features = ["age", "resting bp s", "cholesterol", "max heart rate", "oldpeak"]

# 4. Korelasyon analizi (sürekli değişkenler için)
def drop_highly_correlated_features(df, threshold=0.85):
    corr_matrix = df[continuous_features].corr()
    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1)
    high_corr = (corr_matrix.where(upper_triangle > 0).stack().sort_values(ascending=False))
    to_drop = [column for column in corr_matrix.columns if any(corr_matrix[column] > threshold)]
    return df.drop(columns=to_drop, errors="ignore"), high_corr

# 5. Düşük etkili sütunların kaldırılması (mutual information ile)
def drop_low_importance_features(df, target_column, threshold=0.01):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    mi = mutual_info_classif(X, y, discrete_features='auto', random_state=0)
    mi_series = pd.Series(mi, index=X.columns)
    selected_features = mi_series[mi_series > threshold].index.tolist()
    return df[selected_features + [target_column]], mi_series.sort_values(ascending=False)

# 6. Veri standardizasyonu
scaler = StandardScaler()


def scale_data(df, continuous_features):
    # Veri çerçevesinde olmayan sütunları kaldır
    valid_features = [col for col in continuous_features if col in df.columns]
    if not valid_features:
        raise ValueError("Sürekli değişkenlerin hiçbiri veri setinde bulunmuyor!")

    df[valid_features] = scaler.fit_transform(df[valid_features])
    return df

# İşlem sırasını uygulama
check_missing_values(df)  # Eksik veri kontrolü
df = remove_duplicates(df)  # Tekrarlayan satırların kaldırılması
df = scale_data(df, continuous_features)  # Ölçeklendirme
df, high_corr = drop_highly_correlated_features(df)  # Korelasyon analizi
df, feature_importance = drop_low_importance_features(df, target_column="target")  # Özellik seçimi


print("İşlenmiş veri seti:\n", df)

# Sonuçları yeni bir CSV dosyasına kaydetme
df.to_csv("processed_dataset.csv", index=False)

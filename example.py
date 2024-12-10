import pandas as pd

# Veri setini yükleyin
df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")

# Aykırı değerlerin işleneceği sütunlar ve koşulları
columns_with_outliers = {
    'cholesterol': 0,          # 0 anlam dışı
    'resting bp s': 0,         # 0 anlam dışı
    'oldpeak': lambda x: x < 0  # Negatif değerler anlam dışı
}

# Her bir sütun için aykırı değerleri medyan ile dolduralım
for column, condition in columns_with_outliers.items():
    if callable(condition):  # Koşul bir fonksiyon ise (örneğin negatif değer kontrolü)
        median_value = df.loc[~df[column].apply(condition), column].median()
        df.loc[df[column].apply(condition), column] = median_value
    else:  # Koşul sabit bir değer ise (örneğin 0 kontrolü)
        median_value = df.loc[df[column] != condition, column].median()
        df.loc[df[column] == condition, column] = median_value

# Güncel veri setinin istatistiklerini inceleyin
print(df.describe())

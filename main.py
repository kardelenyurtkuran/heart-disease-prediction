import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Veri setini yükleme
df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")

# Genel bilgi ve özet
print(df.info())

# Sütun isimleri
print(df.columns)

# Eksik değer kontrolü
print(df.isnull().sum())

# Özet istatistikler
print(df.describe())

# Veri tipleri ve eşsiz değer sayısı
print(df.dtypes)
print(df.nunique())

# Hedef değişkenin dağılımı
sns.countplot(x="target", data=df)
plt.show()

# Göğüs ağrısı türünü (chest pain type) one-hot encode
df_encoded = pd.get_dummies(df, columns=['chest pain type'], prefix='cp', drop_first=True)

# Sayısal sütunlar
numerical_cols = ['resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']

# StandardScaler kullanımı
scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

# Sayısal sütunlar için istatistiksel özet
print(df_encoded[numerical_cols].describe())

# Eksik veri kontrolü
print(df_encoded.isnull().sum())




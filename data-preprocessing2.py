import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Veri setini yükleyin (örnek veriyi df olarak kabul ediyorum)
df = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')

# 1. Kategorik ve sayısal sütunları ayırma
categorical_columns = ['sex', 'chest pain type', 'resting ecg', 'exercise angina', 'ST slope']
numerical_columns = ['age', 'resting bp s', 'cholesterol', 'fasting blood sugar', 'max heart rate', 'oldpeak']

# 2. Pipeline ile preprocessing
# OneHotEncoder kategorik değişkenleri dönüştürür, StandardScaler sayısal veriyi ölçeklendirir
categorical_transformer = OneHotEncoder(drop='first')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ]
)

# Preprocessing pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Veriyi dönüştür
processed_data = pipeline.fit_transform(df)

# 3. PCA ile boyut azaltma (opsiyonel)
pca = PCA(n_components=2)  # İlk 2 ana bileşeni kullanarak boyut azaltma
reduced_data = pca.fit_transform(processed_data)

# 4. Sonuçları bir DataFrame'e aktar
processed_df = pd.DataFrame(processed_data.toarray() if hasattr(processed_data, 'toarray') else processed_data)
reduced_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])

# İşlenmiş veri setini CSV olarak kaydet
processed_df.to_csv('cleaned_dataset.csv', index=False)

# PCA ile boyut azaltılmış veri setini CSV olarak kaydet
reduced_df.to_csv('pca_reduced_dataset.csv', index=False)

print("Veriler başarıyla kaydedildi!")


# Çıktıları kontrol et
print("Preprocessed DataFrame Head:\n", processed_df.head())
print("Reduced DataFrame Head:\n", reduced_df.head())

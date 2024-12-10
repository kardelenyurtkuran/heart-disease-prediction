import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Veri setini yükleme
df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")

#------------------Veri Yapısının İncelenmesi-----------------------------------------

# print(df.head()) #veriyi incele
# print(df.info()) #veriyi incele, type
# print(df.describe()) #istatistik değerleri incele

#-------------------Missing Data Handling----------------------------------------------

# print(df.dropna()) #eksik satırları kaldır
# print(df.dropna([1,2,3])) #eksik değerler için belirli satırları kaldır
# print(df.isnull().sum()) #toplam ekisk değer sayısı
# print(df.isnull().sum(axis=1)) #satır bazında eksik değerler
# print(df.isnull().mean()*100) # eksik satırların oranını bulmak
# print(df.isna().sum()) #issnull ile aynı işlev
# print(df.dropna(subset=["age"])) #age sütununda eksik olan satırları kaldırmak
# print(df.dropna(tresh=2)) #bir satırda ne az 2 eksik değer varsa o satır silinir (treshold)

#-------------------Data Transformation------------------------------------------------
# Kategorik tür dönüşümleri
# categorical_cols = ["sex", "chest pain type", "fasting blood sugar",
#                     "resting ecg", "exercise angina", "ST slope", "target"]
#
# # Dönüşüm işlemi
# df[categorical_cols] = df[categorical_cols].astype("category")
#
# # print(df.dtypes) #kontrol et
# print(df.dtypes)
 #-------------------Split train and test-----------------------------------------------

X= df.drop('target', axis=1) #target hedef değişken olduğu için bu sütun çıkarılır
y = df['target'] #hedef değişken
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #%20'si test verisidir, 42de başlangıç değeridir ve otostopcunun galaksi rehberinde 42 sayısı herşeyin yanıtı olarak belirlenmiştir semboliktir herhangi bir sayı kullanılabilir.
#random_state kullanmazsan her çalıştırmada rasgele seçilir bu nedenle farklı olur

print("Eğitim verisi:", X_train.shape, y_train.shape) #952 nesne, 11 öznitelik
print("Test verisi:", X_test.shape, y_test.shape) #238 nesne, 11 öznitelik

# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42) #eğer dengesiz bir dağılım varsa eğitim ve test verisinde ki target oranını eşitlemek için stratify parametresini kullanabilirsin
# print(y_train.value_counts()) # dengesizliği kontrol et
# print(y_test.value_counts()) # dengesizliği kontrol et
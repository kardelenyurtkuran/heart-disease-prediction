import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Veri setini yükleme
df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")

#------------------Veri Yapısının İncelenmesi-----------------------------------------

print(df.head()) #veriyi incele
print(df.info()) #veriyi incele, type
print(df.describe()) #istatistik değerleri incele
print(df.isnull().sum()) #eksik veri kontrolü
print(df['target'].value_counts()) #sınıf dengesini kontrol et
sns.histplot(df['age'], kde=True, color='blue', label='Original Age') #ölçeklendirme öncesi veri


#------------------Yinelenen verilerin kaldırılması-----------------------------------
duplicates = df.duplicated()
print(duplicates.sum())
df = df.drop_duplicates()
print(df.info())

#-------------------Aykırı değer analizi----------------------------------------------

#sürekli değişkenler için
sns.boxplot(data=df[['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']])
plt.show()

#-------------------Kategorik verilerin dönüşümü--------------------------------------

#nominal kategorik veriler için
df = pd.get_dummies(df, columns=['chest pain type', 'resting ecg', 'ST slope'], drop_first=True)
print(df.info()) # sütunlarda ki değişimi kontrol edebilirsin

#------------------Sürekli verilerin ölçeklendirilmesi--------------------------------

scaler = StandardScaler()
df[['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']] = scaler.fit_transform(df[['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']])
sns.histplot(df['age'], kde=True, color='red', label='Scaled Age') #ölçeklendirme sonrası
plt.legend()
plt.show()

#----------------Sürekli değişkenler için korelasyon analizi--------------------------

corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

#----------------Eğitim ve test verilerini ayırma--------------------------------------

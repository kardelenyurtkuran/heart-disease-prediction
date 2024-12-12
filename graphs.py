import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import numpy as np
df_cleaned = pd.read_csv("cleaned_scaled_heart_disease_data.csv")


#---------------------Sınıf dağılımı grafiği------------------------
# print(df_cleaned.info())
# print(df_cleaned['target'].value_counts())
# sns.countplot(x='target', data=df_cleaned, palette='pastel')
# plt.title('Target Değişkeninin Sınıf Dağılımı')
# plt.show()


#-------------------Sürekli değişkenler için aykırı değer-----------
# sns.boxplot(data=df_cleaned[['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']])
# plt.show()


#--------------------Cinsiyet-kolesterol Dağılımı-------------------

# sns.boxplot(x='sex', y='cholesterol', data=df_cleaned)
# plt.title('Cinsiyet ve Kolesterol Dağılımı')
# plt.show()

#--------------------Yaş-Kalp Hastalığı------------------------------

# sns.boxplot(x='target', y='age', data=df_cleaned)
# plt.title('Yaş ve Kalp Hastalığı İlişkisi')
# plt.show()

#---------------------Hedef Değişken-Yaş----------------------------

# sns.histplot(df_cleaned['age'], kde=True, color='blue', label='Age Distribution')
# plt.title('Yaş ve Kalp Hastalığı Dağılımı')
# plt.show()


#---------------------Kolesterol- Hedef Değişken-------------------

# sns.scatterplot(x='cholesterol', y='target', data=df_cleaned, color='green')
# plt.title('Cholesterol ve Kalp Hastalığı İlişkisi')
# plt.show()

#------------------Sürekli değişkenler arasında ki ilişki----------

sns.pairplot(df_cleaned[['age', 'cholesterol', 'max heart rate', 'oldpeak', 'target']], hue='target')
plt.title('Değişkenler Arası İlişkiler')
plt.show()



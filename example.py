import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")
# Cinsiyet ve Kolesterol ilişkisini görselleştir
# plt.figure(figsize=(8, 6))
# sns.boxplot(x='sex', y='cholesterol', data=df)
# plt.title('Cinsiyet ve Kolesterol İlişkisi')
# plt.xlabel('Cinsiyet (0: Kadın, 1: Erkek)')
# plt.ylabel('Kolesterol (mg/dL)')
# plt.show()

# Yaş ve Kolesterol ilişkisini görselleştir
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x='age', y='cholesterol', data=df)
# plt.title('Yaş ve Kolesterol İlişkisi')
# plt.xlabel('Yaş')
# plt.ylabel('Kolesterol (mg/dL)')
# plt.show()

# Cinsiyet ve Kalp Hastalığı ilişkisini görselleştir
# plt.figure(figsize=(8, 6))
# sns.countplot(x='sex', hue='target', data=df)
# plt.title('Cinsiyet ve Kalp Hastalığı İlişkisi')
# plt.xlabel('Cinsiyet (0: Kadın, 1: Erkek)')
# plt.ylabel('Hastaların Sayısı')
# plt.legend(title='Kalp Hastalığı', labels=['Yok', 'Var'])
# plt.show()

# Yaş ve Kalp Hastalığı ilişkisini görselleştir
plt.figure(figsize=(8, 6))
sns.boxplot(x='target', y='age', data=df)
plt.title('Yaş ve Kalp Hastalığı İlişkisi')
plt.xlabel('Kalp Hastalığı Durumu (0: Yok, 1: Var)')
plt.ylabel('Yaş')
plt.show()




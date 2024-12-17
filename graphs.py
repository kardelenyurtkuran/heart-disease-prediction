import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')

# Grafik 1: ST Segmenti Eğimi ve Hedef Dağılımı (Stacked Bar Plot)
st_slope_target = df.groupby(["ST slope", "target"]).size().unstack()
st_slope_target.plot(kind="bar", stacked=True, figsize=(8, 6), color=["#1f77b4", "#ff7f0e"])
plt.title("ST Slope vs. Target (Stacked)")
plt.xlabel("ST Slope")
plt.ylabel("Count")
plt.legend(["No Disease (0)", "Disease (1)"], title="Target")
plt.grid(axis="y")
plt.show()

# Grafik 2: Eski Puan (Oldpeak) ve Hedef İlişkisi (Box Plot)
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="target", y="oldpeak", palette="coolwarm")
plt.title("Oldpeak vs. Target")
plt.xlabel("Target (0: No Disease, 1: Disease)")
plt.ylabel("Oldpeak")
plt.grid(True)
plt.show()

# Grafik 3: Cinsiyet ve Hedef Dağılımı (Bar Plot)
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x="sex", hue="target", palette="Set2")
plt.title("Gender vs. Target")
plt.xlabel("Sex (0: Female, 1: Male)")
plt.ylabel("Count")
plt.legend(title="Target", loc="upper right")
plt.grid(axis="y")
plt.show()

# Grafik 4: Egzersiz Anjinası ve Maksimum Kalp Hızı (Box Plot)
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="exercise angina", y="max heart rate", palette="muted")
plt.title("Exercise Angina vs. Max Heart Rate")
plt.xlabel("Exercise Angina (0: No, 1: Yes)")
plt.ylabel("Max Heart Rate")
plt.grid(True)
plt.show()

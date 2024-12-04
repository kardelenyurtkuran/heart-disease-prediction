import pandas as pd

# Veri setini yükleme
df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")

# Genel bilgiler
print(df.info())
print(df.head())



from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

def encode_boolean_columns(df, threshold=-1):
    for column in df.columns:
        # Eğer değişken sürekli bir değişkense ve negatif değerler varsa, 0/1'e çevirin
        if df[column].dtype in ['float64', 'int64'] and (df[column] < 0).any():
            df[column] = (df[column] >= threshold).astype(int)
    return df

# Veriyi güncellenmiş olarak okuyun
df = pd.read_csv('cleaned_scaled_heart_disease_data.csv')
df = encode_boolean_columns(df)


from mlxtend.frequent_patterns import apriori, association_rules

# Min support değeri
min_support = 0.03  # %3

# Apriori algoritması
frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True, verbose=1)

# İlişkilendirme kuralları oluşturma
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0, num_itemsets=10)

# Kuralları filtreleme
filtered_rules = rules[rules['confidence'] >= 0.7]

# En ilginç 15 kuralı göster
top_15_rules = filtered_rules.nlargest(15, 'lift')

# Kuralları yazdır
print("Top 15 Interesting Association Rules:\n")
for index, row in top_15_rules.iterrows():
    print(f"Rule: {row['antecedents']} => {row['consequents']} | Support: {row['support']:.2f} | Confidence: {row['confidence']:.2f} | Lift: {row['lift']:.2f}")

df_rules = pd.DataFrame(rules)
df_rules.to_csv("rules_output.csv", index=False)
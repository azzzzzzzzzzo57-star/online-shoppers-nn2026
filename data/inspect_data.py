
import pandas as pd

df = pd.read_csv("data/raw/online_shoppers.csv")

print("\n1. Dataset Shape:")
print(df.shape)

print("\n2. Data Types:")
print(df.dtypes)

print("\n3. Missing Values:")
print(df.isna().sum().sum())

print("\n4. Duplicate Rows:")
print(df.duplicated().sum())

print("\n5. Revenue Distribution:")
print(df["Revenue"].value_counts())

print("\n6. Revenue Percentages:")
print(df["Revenue"].value_counts(normalize=True) * 100)

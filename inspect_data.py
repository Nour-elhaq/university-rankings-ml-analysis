import pandas as pd
import numpy as np

# Load the data
df = pd.read_excel('Top_Universities_THE.xlsx')

print("=" * 80)
print("DATA STRUCTURE AND CONTENT INSPECTION")
print("=" * 80)

print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")

print("\n" + "-" * 80)
print("COLUMNS:")
print("-" * 80)
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print("\n" + "-" * 80)
print("DATA TYPES:")
print("-" * 80)
print(df.dtypes)

print("\n" + "-" * 80)
print("MISSING VALUES:")
print("-" * 80)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0])

print("\n" + "-" * 80)
print("FIRST 10 ROWS:")
print("-" * 80)
print(df.head(10))

print("\n" + "-" * 80)
print("BASIC STATISTICS:")
print("-" * 80)
print(df.describe())

print("\n" + "-" * 80)
print("UNIQUE VALUES IN KEY COLUMNS:")
print("-" * 80)
for col in df.columns:
    unique_count = df[col].nunique()
    if unique_count < 50:  # Only show for columns with less than 50 unique values
        print(f"\n{col}: {unique_count} unique values")
        print(df[col].value_counts().head(10))

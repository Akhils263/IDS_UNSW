#Exploratory Data Analysis

import pandas as pd 

train = pd.read_parquet('datasets/UNSW_NB15_training-set.parquet')
test  = pd.read_parquet('datasets/UNSW_NB15_testing-set.parquet')


print("="*50)
print("📊 Dataset Shapes")
print("="*50)
print(f"Train shape: {train.shape}")
print(f"Test shape : {test.shape}")

print("\n" + "="*50)
print("📋 Columns")
print("="*50)
print(train.columns.tolist())

print("\n" + "="*50)
print("🔎 Sample Rows (train)")
print("="*50)
print(train.head(5).to_string(index=False))  # avoids cutting columns

print("\n" + "="*50)
print("ℹ️ Column Info (train)")
print("="*50)
print(train.info())

print("\n" + "="*50)
print("❓ Missing Values (train)")
print("="*50)
missing = train.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))

print("\n" + "="*50)
print("📂 Label Distribution (attack categories)")
print("="*50)
print(train['attack_cat'].value_counts())

print("\n" + "="*50)
print("⚔️ Binary Label Distribution (0=Normal, 1=Attack)")
print("="*50)
print(train['label'].value_counts())

print("\n✅ EDA script finished successfully!")
input("Press Enter to close...")
import os
import pandas as pd
import numpy as np

RAW_FILE = "data/raw/processed.cleveland.data"
OUT_FILE = "data/cleaned/heart_clean.csv"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num"
]

# Load raw data
df = pd.read_csv(RAW_FILE, header=None, names=COLUMNS, na_values='?')

# Convert to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Convert target to binary
df["target"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
df.drop(columns=["num"], inplace=True)

# Drop rows with missing values
df_clean = df.dropna().reset_index(drop=True)

# Save cleaned CSV
os.makedirs("data/cleaned", exist_ok=True)
df_clean.to_csv(OUT_FILE, index=False)

print("Clean dataset saved to:", OUT_FILE)
print("Final shape:", df_clean.shape)

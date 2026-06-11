import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

input_path = "data/external"
output_path = "data/processed/processed.csv"

csv_file = None
for root, dirs, files in os.walk(input_path):
    for file in files:
        if file.endswith(".csv"):
            csv_file = os.path.join(root, file)
            break

if csv_file is None:
    raise ValueError("No CSV found in extracted data")

df = pd.read_csv(csv_file)

print("\n=== Dataset Info BEFORE Processing ===")
print(df.dtypes)
print("\nNumber of rows:", len(df))

if 'isFraud' not in df.columns:
    raise ValueError("Column 'isFraud' not found in dataset")

print("\nClass distribution (RAW):")
print(df['isFraud'].value_counts())

df = df.drop(columns=["nameOrig", "nameDest"], errors='ignore')

df["balance_diff_org"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

if 'type' in df.columns:
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n=== Dataset Info AFTER Processing ===")
print(df.dtypes)
print("\nNumber of rows:", len(df))

print("\nClass distribution (FINAL):")
print(df['isFraud'].value_counts())

os.makedirs("data/processed", exist_ok=True)
df.to_csv(output_path, index=False)

print("\nPreprocessing done! Saved at:", output_path)
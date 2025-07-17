import pandas as pd

def load_csv(file_path):
    return pd.read_csv(file_path)

def save_csv(df, file_path):
    df.to_csv(file_path, index=False)
    print(f"✅ Data saved to {file_path}")
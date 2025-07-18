import pandas as pd
import os


def clean_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop_duplicates()
    df = df.fillna(df.mean(numeric_only=True))
    return df


if __name__ == "__main__":
    raw_data_path = "../notebooks/data/sample_data.csv"
    output_path = "../notebooks/data/cleaned_data.csv"

    if os.path.exists(raw_data_path):
        cleaned_df = clean_data(raw_data_path)
        cleaned_df.to_csv(output_path, index=False)
        print(f"✅ Cleaned data saved to {output_path}")
    else:
        print("❌ Raw data file not found!")
import pandas as pd
from datetime import datetime
import os

def append_timestamp(file_path):
    df = pd.read_csv(file_path)
    df['updated_at'] = datetime.now().isoformat()
    return df

if __name__ == "__main__":
    data_file = "../notebooks/data/cleaned_data.csv"

    if os.path.exists(data_file):
        updated_df = append_timestamp(data_file)
        updated_df.to_csv(data_file, index=False)
        print(f"✅ Dataset updated with timestamp in {data_file}")
    else:
        print("❌ Cleaned data file not found!")

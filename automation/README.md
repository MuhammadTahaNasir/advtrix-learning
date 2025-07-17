# 🛠️ Automation Scripts

This folder contains automation and data generation scripts to support the Advtrix Learning project's data-driven notebooks and applications.

## Available Scripts

| Script                   | Description                          |
|--------------------------|--------------------------------------|
| generate_churn_data.py   | Generates churn dataset for analysis (output: `notebooks/data/churn_data.csv`) |
| generate_sample_data.py  | Creates leads, customers, and marketing trends datasets in `notebooks/data/` |
| data_cleaner.py          | Cleans any CSV dataset by removing duplicates and filling missing numeric values |
| update_dataset.py        | Appends a timestamp column to datasets |
| automation_helper.py     | Utility functions for loading and saving CSV files |

## Usage

```bash
# Generate churn dataset
python generate_churn_data.py

# Generate all sample datasets
python generate_sample_data.py

# Clean any dataset (edit paths inside script)
python data_cleaner.py

# Update dataset with timestamp
python update_dataset.py

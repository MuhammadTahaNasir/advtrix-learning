import pandas as pd
import numpy as np
import os

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

np.random.seed(42)
n_samples = 1000

data = {
    'customer_id': range(1, n_samples + 1),
    'tenure_months': np.random.randint(1, 60, n_samples),
    'monthly_spend': np.random.uniform(20, 200, n_samples),
    'support_tickets': np.random.randint(0, 10, n_samples),
    'churned': np.random.choice([0, 1], n_samples, p=[0.75, 0.25])
}

df = pd.DataFrame(data)
df.to_csv('data/churn_data.csv', index=False)
print("✅ Churn data saved to data/churn_data.csv")

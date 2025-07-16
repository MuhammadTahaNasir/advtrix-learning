import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()

# ----------------------------
# 1. Generate Leads Data
sources = ['LinkedIn', 'Facebook', 'Email', 'Google Ads', 'Referral']
industries = ['Tech', 'Healthcare', 'Finance', 'Education', 'Retail']

leads = []
for i in range(1, 101):
    leads.append({
        'lead_id': i,
        'source': random.choice(sources),
        'industry': random.choice(industries),
        'monthly_income': random.randint(3000, 9000),
        'previous_interactions': random.randint(0, 10),
        'converted': random.choice([0, 1])
    })

leads_df = pd.DataFrame(leads)
leads_df.to_csv('notebooks/data/leads_data.csv', index=False)
print("✅ leads_data.csv generated with 100 records")

# ----------------------------
# 2. Generate Customers Data
segments = ['Silver', 'Gold', 'Platinum']
genders = ['M', 'F']

customers = []
for i in range(101, 201):
    income = random.randint(3000, 10000)
    if income > 7000:
        segment = 'Platinum'
    elif income > 5000:
        segment = 'Gold'
    else:
        segment = 'Silver'
    
    customers.append({
        'customer_id': i,
        'age': random.randint(18, 65),
        'income': income,
        'gender': random.choice(genders),
        'purchase_frequency': random.randint(1, 12),
        'segment': segment
    })

customers_df = pd.DataFrame(customers)
customers_df.to_csv('notebooks/data/customers_data.csv', index=False)
print("✅ customers_data.csv generated with 100 records")

# ----------------------------
# 3. Generate Marketing Trends Data
channels = ['Facebook', 'Google Ads', 'LinkedIn', 'Email', 'Referral']
start_date = datetime(2023, 1, 1)

trends = []
for i in range(100):
    month = (start_date + timedelta(days=30*i)).strftime('%Y-%m')
    channel = random.choice(channels)
    spend = random.randint(1000, 5000)
    conversions = random.randint(50, 300)
    
    trends.append({
        'month': month,
        'channel': channel,
        'spend': spend,
        'conversions': conversions
    })

trends_df = pd.DataFrame(trends)
trends_df.to_csv('notebooks/data/marketing_trends.csv', index=False)
print("✅ marketing_trends.csv generated with 100 records")

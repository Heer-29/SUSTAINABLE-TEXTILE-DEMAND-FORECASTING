# create_sample_data.py - Creates sample data for testing your Streamlit app
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("📊 Creating Sample Textile Sales Data...")

# Set random seed
np.random.seed(42)

# Generate sample data
start_date = datetime(2024, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(100)]

data = []
skus = ['SKU001', 'SKU002', 'SKU003', 'SKU004', 'SKU005']
stores = ['Store_A', 'Store_B', 'Store_C', 'Store_D']
materials = ['organic_cotton', 'recycled_polyester', 'hemp', 'bamboo', 'linen', 'cotton', 'polyester']
categories = ['t-shirt', 'jeans', 'dress', 'jacket', 'shorts', 'sweater']

for date in dates:
    # Generate 2-3 records per day
    for _ in range(np.random.randint(2, 4)):
        sku = np.random.choice(skus)
        store = np.random.choice(stores)
        material = np.random.choice(materials)
        category = np.random.choice(categories)
        
        # Generate realistic sales based on seasonality
        base_sales = 100
        if date.month in [3, 4, 5, 6, 7, 8]:  # Spring/Summer
            base_sales *= 1.2
        if date.month in [11, 12]:  # Holiday season
            base_sales *= 1.4
        
        # Add some randomness
        units = int(base_sales + np.random.normal(0, 30))
        units = max(20, min(units, 300))  # Keep reasonable bounds
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'sku': sku,
            'store': store,
            'units': units,
            'material': material,
            'category': category
        })

# Create DataFrame and save
df = pd.DataFrame(data)
df.to_csv('sample_data.csv', index=False)

print(f"✅ Created {len(df)} sample records")
print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
print(f"📦 Unique SKUs: {df['sku'].nunique()}")
print(f"🏪 Stores: {', '.join(df['store'].unique())}")
print(f"📊 Average daily sales: {df['units'].mean():.0f} units")

print(f"\n📋 Sample data preview:")
print(df.head(10))
print(f"\n💾 Sample data saved as 'sample_data.csv'")

import pandas as pd
import numpy as np

# === Config ===
start_date = "2022-01-01"
end_date = "2023-12-31"
skus = [f"SKU_{i+1:03d}" for i in range(20)]  # 20 SKUs
categories = ["Shirts", "Pants", "Jackets", "Accessories"]
materials = ["Organic Cotton", "Recycled Polyester", "Linen", "Hemp"]
recycled_content = {
    "Organic Cotton": 0.2,
    "Recycled Polyester": 0.9,
    "Linen": 0.1,
    "Hemp": 0.3,
}

# === Generate dates ===
date_rng = pd.date_range(start=start_date, end=end_date, freq="D")
np.random.seed(42)

data = []
for date in date_rng:
    for sku in np.random.choice(skus, size=5, replace=False):  # 5 SKUs per day
        cat = np.random.choice(categories)
        mat = np.random.choice(materials)
        rec_content = recycled_content[mat]
        price = np.random.choice([20, 30, 40, 50])

        # Base demand with seasonality
        base_demand = 200 + 15 * np.sin(2 * np.pi * date.dayofyear / 365)

        # Trend (higher demand in 2023)
        trend = (date.year - 2022) * 10

        # Promotion effect (May & November)
        promo_flag = 1 if date.month in [5, 11] else 0
        promo_effect = 20 if promo_flag else 0

        # Sustainability campaign effect (all of 2023)
        eco_campaign_flag = 1 if date.year >= 2023 else 0
        eco_effect = 10 if eco_campaign_flag else 0

        # Random noise
        noise = np.random.normal(0, 5)

        demand = round(base_demand + trend + promo_effect + eco_effect + noise, 0)

        data.append([
            date, sku, cat, mat, rec_content, price,
            promo_flag, eco_campaign_flag, demand
        ])

# === Create DataFrame ===
df = pd.DataFrame(data, columns=[
    "date", "sku", "category", "material", "recycled_content",
    "price", "promo_flag", "eco_campaign_flag", "demand"
])

# === Save to CSV ===
df.to_csv("sustainable_textile_demand.csv", index=False)

print("✅ sustainable_textile_demand.csv generated successfully!")
print(df.head(10))

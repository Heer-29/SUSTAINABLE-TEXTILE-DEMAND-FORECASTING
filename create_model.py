# create_model.py - Creates your trained sustainable textile forecasting model
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from datetime import datetime, timedelta

print("🔧 Creating Sustainable Textile Demand Forecasting Model...")
print("=" * 60)

# Set random seed for reproducibility
np.random.seed(42)

def generate_textile_data(n_samples=5000):
    """Generate realistic textile demand data with seasonal patterns"""
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    total_days = (end_date - start_date).days
    
    data = []
    
    for i in range(n_samples):
        # Random date
        random_days = np.random.randint(0, total_days)
        date = start_date + timedelta(days=random_days)
        
        # Basic temporal features
        weekofyear = int(date.isocalendar().week)
        month = int(date.month)
        quarter = int((month - 1) // 3 + 1)
        day_of_week = int(date.weekday())
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Seasonal features
        is_holiday_season = 1 if month in [11, 12] else 0
        is_spring_summer = 1 if month in [3, 4, 5, 6, 7, 8] else 0
        
        # Sustainability and category indicators
        is_sustainable = np.random.choice([0, 1], p=[0.6, 0.4])
        is_seasonal_category = np.random.choice([0, 1], p=[0.8, 0.2])
        
        # Lag features (simulated historical sales)
        base_lag = np.random.normal(120, 30)
        units_lag_1 = max(10, base_lag + np.random.normal(0, 10))
        units_lag_2 = max(10, base_lag + np.random.normal(0, 15))
        units_lag_4 = max(10, base_lag + np.random.normal(0, 20))
        units_lag_8 = max(10, base_lag + np.random.normal(0, 25))
        
        # Rolling averages
        units_ma_4 = max(10, base_lag + np.random.normal(0, 12))
        units_ma_8 = max(10, base_lag + np.random.normal(0, 18))
        units_ma_12 = max(10, base_lag + np.random.normal(0, 22))
        
        # Rolling standard deviations
        units_std_4 = max(1, np.random.normal(15, 5))
        units_std_8 = max(1, np.random.normal(18, 6))
        units_std_12 = max(1, np.random.normal(20, 7))
        
        # Calculate realistic demand
        base_demand = 80
        seasonal_multiplier = 1.0
        
        if is_spring_summer:
            seasonal_multiplier *= 1.2
        if is_holiday_season:
            seasonal_multiplier *= 1.4
        if is_weekend:
            seasonal_multiplier *= 1.1
        if is_sustainable:
            seasonal_multiplier *= 1.15
        
        # Month-specific patterns
        month_effects = {1: 0.8, 2: 0.9, 3: 1.1, 4: 1.2, 5: 1.3, 6: 1.2,
                        7: 1.4, 8: 1.3, 9: 1.1, 10: 1.0, 11: 1.3, 12: 1.5}
        seasonal_multiplier *= month_effects.get(month, 1.0)
        
        # Historical momentum effect
        momentum = (units_lag_1 * 0.4 + units_lag_2 * 0.3 + 
                   units_lag_4 * 0.2 + units_lag_8 * 0.1) * 0.01
        
        # Calculate final demand
        demand = (base_demand * seasonal_multiplier + momentum + 
                 np.random.normal(0, 15))
        
        demand = max(15, min(demand, 500))
        
        data.append({
            'weekofyear': weekofyear,
            'month': month,
            'quarter': quarter,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'is_holiday_season': is_holiday_season,
            'is_spring_summer': is_spring_summer,
            'is_sustainable': is_sustainable,
            'is_seasonal_category': is_seasonal_category,
            'units_lag_1': units_lag_1,
            'units_lag_2': units_lag_2,
            'units_lag_4': units_lag_4,
            'units_lag_8': units_lag_8,
            'units_ma_4': units_ma_4,
            'units_ma_8': units_ma_8,
            'units_ma_12': units_ma_12,
            'units_std_4': units_std_4,
            'units_std_8': units_std_8,
            'units_std_12': units_std_12,
            'units': demand
        })
    
    return pd.DataFrame(data)

# Generate and train model
print("📊 Generating realistic textile demand data...")
df = generate_textile_data(5000)

print(f"✅ Generated {len(df)} training samples")
print(f"📈 Average demand: {df['units'].mean():.1f} units")
print(f"🌱 Sustainable products: {(df['is_sustainable'].sum() / len(df) * 100):.1f}%")

# Prepare features and target
feature_columns = [col for col in df.columns if col != 'units']
X = df[feature_columns]
y = df['units']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("\n🌳 Training Random Forest Model...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate model
test_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, test_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"📊 Model Performance:")
print(f"   MAE: {test_mae:.2f}")
print(f"   R²:  {test_r2:.3f}")
print(f"   Accuracy: {(100 - test_mae/y_test.mean()*100):.1f}%")

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"\n✅ Model saved as 'model.pkl'")
print(f"🎯 Your sustainable textile forecasting model is ready!")

# Save feature names for reference
with open('model_features.txt', 'w') as f:
    f.write("Features expected by the model:\n")
    for i, feature in enumerate(feature_columns, 1):
        f.write(f"{i:2d}. {feature}\n")

print(f"📋 Feature list saved as 'model_features.txt'")

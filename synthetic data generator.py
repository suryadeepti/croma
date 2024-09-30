import numpy as np
import pandas as pd
import random

#values for each feature
regions = ['North', 'South', 'East', 'West']
product_categories = ['TV', 'Washing Machine', 'Refrigerator', 'Air Conditioner']
promotional_period = [0, 1]  # 0 for no promotion, 1 for promotion
weather_conditions = ['Sunny', 'Cloudy', 'Rainy']

#synthetic data for 1000 rows
data_size = 1000
synthetic_data = []

for _ in range(data_size):
    region = random.choice(regions)
    product_category = random.choice(product_categories)
    promotion = random.choice(promotional_period)
    weather = random.choice(weather_conditions)
    
    # Generate sales quantity based on conditions
    if promotion:
        base_sales = np.random.randint(100, 200)
    else:
        base_sales = np.random.randint(50, 150)
    
    # Adjusting sales based on weather and region
    if weather == 'Sunny':
        base_sales += 20
    if region == 'North':
        base_sales += 10

    sales_quantity = base_sales
    
    synthetic_data.append([region, product_category, promotion, weather, sales_quantity])

# Convert to DataFrame
df = pd.DataFrame(synthetic_data, columns=['region', 'product_category', 'promotional_period', 'weather_conditions', 'sales_quantity'])

# Display the first few rows
print(df.head())
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Encode categorical variables
label_encoders = {}
for col in ['region', 'product_category', 'weather_conditions']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

scaler = MinMaxScaler()
df['sales_quantity'] = scaler.fit_transform(df[['sales_quantity']])

X = df.drop('sales_quantity', axis=1)
y = df['sales_quantity']

X = X.values
y = y.values
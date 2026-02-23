import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load Titanic dataset
file_path = 'datasets/titanic/train.csv'
df = pd.read_csv(file_path)

# Select numeric features for scaling
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']

# Handle missing values
df_numeric = df[numeric_features].copy()
df_numeric = df_numeric.fillna(df_numeric.mean())

# Basic Normalization (Min-Max Scaling)
minmax_scaler = MinMaxScaler()
df_normalized = pd.DataFrame(minmax_scaler.fit_transform(df_numeric), columns=numeric_features)

print('--- Normalized Features (first 5 rows) ---')
print(df_normalized.head())

# Standardization (Z-score)
standard_scaler = StandardScaler()
df_standardized = pd.DataFrame(standard_scaler.fit_transform(df_numeric), columns=numeric_features)

print('\n--- Standardized Features (first 5 rows) ---')
print(df_standardized.head())

# Compare original, normalized, and standardized values for first 5 rows
print('\n--- Original vs Normalized vs Standardized (first 5 rows) ---')
comparison = pd.concat([
    df_numeric.head().reset_index(drop=True),
    df_normalized.head().add_suffix('_norm'),
    df_standardized.head().add_suffix('_std')
], axis=1)
print(comparison)

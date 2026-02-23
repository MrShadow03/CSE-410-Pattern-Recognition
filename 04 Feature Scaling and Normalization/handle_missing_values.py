import pandas as pd

dataset_path = 'datasets/titanic/train.csv'
df = pd.read_csv(dataset_path)

# Check for missing values
print(df.isnull().sum())

df_numeric = df[['Age', 'SibSp', 'Parch', 'Fare']].copy()
df_numeric = df_numeric.fillna(df_numeric.mean())

print(df_numeric.isnull().sum())

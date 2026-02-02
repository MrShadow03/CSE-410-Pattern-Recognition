# ============================================================
# WORKING WITH DATA FOR PATTERN RECOGNITION
# ============================================================
# Goal: Store and manage feature data using lists, NumPy arrays, and Pandas DataFrames
# Outcome: Student can handle data using basic Pandas

print("=" * 60)
print("LESSON 1: LISTS - Basic Feature Data Storage")
print("=" * 60)

# 1.1 Creating Lists
print("\n1.1 Creating Lists:")
features_list = [22, 38, 26, 35]  # Age feature from Titanic
print(f"Ages: {features_list}")

passenger_data = ["Owen", "Florence", "Laina", "Lily"]
print(f"Names: {passenger_data}")

mixed_data = [1, "John", 22.5, True]
print(f"Mixed data: {mixed_data}")

# 1.2 Accessing List Elements
print("\n1.2 Accessing List Elements:")
print(f"First age: {features_list[0]}")
print(f"Last age: {features_list[-1]}")
print(f"Ages 1-2: {features_list[0:2]}")

# 1.3 Modifying Lists
print("\n1.3 Modifying Lists:")
features_list[0] = 23
print(f"Updated first age: {features_list}")

features_list.append(40)
print(f"After append: {features_list}")

features_list.extend([50, 55])
print(f"After extend: {features_list}")

features_list.remove(23)
print(f"After remove(23): {features_list}")

# 1.4 List Methods
print("\n1.4 Common List Methods:")
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
print(f"Original: {numbers}")
print(f"Length: {len(numbers)}")
print(f"Max: {max(numbers)}, Min: {min(numbers)}")
print(f"Sum: {sum(numbers)}")
print(f"Count of 1s: {numbers.count(1)}")
numbers.sort()
print(f"Sorted: {numbers}")

print("\n" + "=" * 60)
print("LESSON 2: NumPy ARRAYS - Efficient Numerical Computing")
print("=" * 60)

import numpy as np

# 2.1 Creating NumPy Arrays
print("\n2.1 Creating NumPy Arrays:")
ages_array = np.array([22, 38, 26, 35, 40])
print(f"Ages array: {ages_array}")
print(f"Type: {type(ages_array)}")
print(f"Data type: {ages_array.dtype}")

fares_array = np.array([7.25, 71.28, 7.92, 53.1, 8.05])
print(f"Fares array: {fares_array}")

# 2.2 Array Properties
print("\n2.2 Array Properties:")
print(f"Shape: {ages_array.shape}")
print(f"Size: {ages_array.size}")
print(f"Dimensions: {ages_array.ndim}")

# 2.3 Creating Arrays with Functions
print("\n2.3 Creating Arrays with Functions:")
zeros_arr = np.zeros(5)
print(f"Zeros: {zeros_arr}")

ones_arr = np.ones(5)
print(f"Ones: {ones_arr}")

range_arr = np.arange(0, 10, 2)
print(f"Range (0 to 10, step 2): {range_arr}")

linspace_arr = np.linspace(0, 10, 5)
print(f"Linspace (0 to 10, 5 points): {linspace_arr}")

# 2.4 Array Operations
print("\n2.4 Array Operations:")
print(f"Ages: {ages_array}")
print(f"Ages + 5: {ages_array + 5}")
print(f"Ages * 2: {ages_array * 2}")
print(f"Ages / 2: {ages_array / 2}")

# 2.5 Array Statistics
print("\n2.5 Array Statistics:")
print(f"Mean age: {np.mean(ages_array):.2f}")
print(f"Std dev: {np.std(ages_array):.2f}")
print(f"Min age: {np.min(ages_array)}")
print(f"Max age: {np.max(ages_array)}")
print(f"Sum of fares: {np.sum(fares_array):.2f}")

# 2.6 2D Arrays (Matrix)
print("\n2.6 2D Arrays (Feature Matrix):")
features_matrix = np.array([
    [22, 7.25, 0],      # Passenger 1: age, fare, survived
    [38, 71.28, 1],     # Passenger 2
    [26, 7.92, 1],      # Passenger 3
    [35, 53.1, 1]       # Passenger 4
])
print(f"Feature Matrix:\n{features_matrix}")
print(f"Shape: {features_matrix.shape}")
print(f"First row (Passenger 1): {features_matrix[0]}")
print(f"First column (Ages): {features_matrix[:, 0]}")

# 2.7 Array Indexing and Slicing
print("\n2.7 Indexing and Slicing:")
print(f"Element at [1, 2]: {features_matrix[1, 2]}")
print(f"First 2 passengers: \n{features_matrix[0:2]}")
print(f"All ages (column 0): {features_matrix[:, 0]}")

print("\n" + "=" * 60)
print("LESSON 3: PANDAS DataFrames - Data Management")
print("=" * 60)

import pandas as pd

# 3.1 Creating a DataFrame
print("\n3.1 Creating a DataFrame:")
data_dict = {
    'PassengerId': [1, 2, 3, 4],
    'Name': ['Owen', 'Florence', 'Laina', 'Lily'],
    'Age': [22, 38, 26, 35],
    'Fare': [7.25, 71.28, 7.92, 53.1],
    'Survived': [0, 1, 1, 1]
}
df = pd.DataFrame(data_dict)
print(df)

# 3.2 DataFrame Info
print("\n3.2 DataFrame Information:")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Data types:\n{df.dtypes}")

# 3.3 Accessing Data
print("\n3.3 Accessing Data:")
print(f"\nAccessing single column 'Age':")
print(df['Age'])

print(f"\nAccessing multiple columns:")
print(df[['Name', 'Age']])

print(f"\nFirst 2 rows:")
print(df.head(2))

print(f"\nLast 2 rows:")
print(df.tail(2))

# 3.4 DataFrame Statistics
print("\n3.4 DataFrame Statistics:")
print(df.describe())

print(f"\nMean age: {df['Age'].mean():.2f}")
print(f"Max fare: {df['Fare'].max():.2f}")
print(f"Survival rate: {df['Survived'].mean():.2%}")

# 3.5 Filtering Data
print("\n3.5 Filtering Data:")
print(f"\nPassengers older than 30:")
print(df[df['Age'] > 30])

print(f"\nPassengers who survived:")
print(df[df['Survived'] == 1])

# 3.6 Adding New Columns
print("\n3.6 Adding New Columns:")
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 20, 40, 100], 
                          labels=['Child', 'Adult', 'Senior'])
print(df)

# 3.7 Loading Real Data
print("\n3.7 Loading Real Titanic Data:")
try:
    titanic_df = pd.read_csv('../datasets/titanic/train.csv')
    print(f"Dataset shape: {titanic_df.shape}")
    print(f"\nFirst 3 rows:")
    print(titanic_df.head(3))
    
    print(f"\nColumn names:")
    print(titanic_df.columns.tolist())
    
    print(f"\nBasic statistics:")
    print(titanic_df[['Age', 'Fare', 'Survived']].describe())
    
    print(f"\nMissing values:")
    print(titanic_df.isnull().sum())
    
except FileNotFoundError:
    print("Titanic dataset not found at ../datasets/titanic/train.csv")

print("\n" + "=" * 60)
print("LESSON 4: PRACTICAL EXERCISES")
print("=" * 60)

# Exercise 1: Feature List
print("\n4.1 Exercise 1 - Feature List:")
print("Task: Create a list of passenger fares and calculate average")
fares = [7.25, 71.28, 7.92, 53.1, 8.05]
avg_fare = sum(fares) / len(fares)
print(f"Fares: {fares}")
print(f"Average fare: ${avg_fare:.2f}")

# Exercise 2: NumPy Feature Matrix
print("\n4.2 Exercise 2 - NumPy Feature Matrix:")
print("Task: Create feature matrix (age, fare, class) and calculate statistics")
features = np.array([
    [22, 7.25, 3],
    [38, 71.28, 1],
    [26, 7.92, 3],
    [35, 53.1, 1],
    [40, 8.05, 3]
])
print(f"Features (Age, Fare, Class):\n{features}")
print(f"Mean age: {np.mean(features[:, 0]):.2f}")
print(f"Mean fare: {np.mean(features[:, 1]):.2f}")
print(f"Passengers in Class 1: {np.sum(features[:, 2] == 1)}")

# Exercise 3: DataFrame Filtering
print("\n4.3 Exercise 3 - DataFrame Filtering:")
print("Task: Filter passengers and calculate survival rate by class")
df_exercise = pd.DataFrame({
    'PassengerId': [1, 2, 3, 4, 5],
    'Age': [22, 38, 26, 35, 40],
    'Pclass': [3, 1, 3, 1, 3],
    'Survived': [0, 1, 1, 1, 0]
})
print(df_exercise)
print(f"\nSurvival rate by class:")
print(df_exercise.groupby('Pclass')['Survived'].mean())

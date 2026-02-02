import numpy as np
import pandas as pd

titanic = {
    'PassengerId': [1, 2, 3, 4],
    'Name': ['Owen', 'Florence', 'Laina', 'Lily'],
    'Age': [22, 38, 26, 35],
    'Fare': [7.25, 71.28, 7.92, 53.1],
    'Survived': [0, 1, 1, 1]
}

df = pd.DataFrame(titanic)

# 3.4 Data age
print(df.describe())
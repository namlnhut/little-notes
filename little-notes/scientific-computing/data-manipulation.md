# Data Manipulation

Pandas provides powerful tools for data manipulation and analysis.

## DataFrames and Series

```python
import pandas as pd
import numpy as np

# Create a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'London', 'Paris', 'Tokyo'],
    'Salary': [70000, 80000, 75000, 85000]
}
df = pd.DataFrame(data)

# Create a Series
s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
```

## Data Selection

```python
# Column selection
print(df['Name'])
print(df[['Name', 'Age']])

# Row selection
print(df.iloc[0])  # By position
print(df.loc[0])   # By label

# Conditional selection
adults = df[df['Age'] > 30]
high_earners = df[df['Salary'] > 75000]
```

## Data Cleaning

```python
# Handle missing values
df_missing = df.copy()
df_missing.loc[1, 'Age'] = np.nan

# Fill missing values
df_filled = df_missing.fillna(df_missing.mean())

# Drop missing values
df_dropped = df_missing.dropna()

# Remove duplicates
df_unique = df.drop_duplicates()
```

## Data Transformation

```python
# Apply functions
df['Age_Squared'] = df['Age'] ** 2
df['Salary_K'] = df['Salary'] / 1000

# Map values
city_codes = {'New York': 'NY', 'London': 'LD', 'Paris': 'PR', 'Tokyo': 'TK'}
df['City_Code'] = df['City'].map(city_codes)

# Apply custom functions
df['Senior'] = df['Age'].apply(lambda x: 'Yes' if x > 30 else 'No')
```

## Grouping and Aggregation

```python
# Group by and aggregate
grouped = df.groupby('City')['Salary'].mean()

# Multiple aggregations
agg_df = df.groupby('City').agg({
    'Age': ['mean', 'min', 'max'],
    'Salary': ['sum', 'mean']
})
```

## Merging and Joining

```python
# Merge DataFrames
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]})

merged = pd.merge(df1, df2, on='key', how='inner')
```

## Reading and Writing Data

```python
# Read from CSV
# df = pd.read_csv('data.csv')

# Write to CSV
# df.to_csv('output.csv', index=False)

# Read from Excel
# df = pd.read_excel('data.xlsx')

# Read from JSON
# df = pd.read_json('data.json')
```

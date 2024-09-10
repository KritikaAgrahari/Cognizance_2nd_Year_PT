#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Print summary statistics
print(df.describe())

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values per feature:\n", missing_values)

# Plot histograms for each feature
df.hist(figsize=(10, 8), bins=30)
plt.show()

# Plot histograms using Seaborn
for column in df.columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

# Plot pairwise relationships
sns.pairplot(df, hue='target')
plt.show()


# In[ ]:





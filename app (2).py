
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

# Membaca data
file_path = 'Clustering.csv'
df = pd.read_csv(file_path)

# 1. sex vs settlement size
sex_vs_settlement_size = df[['Sex', 'Settlement_size']].groupby('Sex', as_index=False).count()
print(sex_vs_settlement_size)

# 2. Marital Status vs Settlement Size
marital_status_vs_settlement_size = df[['Marital_status', 'Settlement_size']].groupby('Marital_status', as_index=False).count()
print(marital_status_vs_settlement_size)

# 3. Age Distribution
age_distribution = df['Age'].describe()
print(age_distribution)

# 4. Income vs Education
income_vs_education = df[['Income', 'Education']].groupby('Education', as_index=False).mean()
print(income_vs_education)

# 5. Occupation vs Income
occupation_vs_income = df[['Occupation', 'Income']].groupby('Occupation', as_index=False).mean()
print(occupation_vs_income)

# 6. Average Income by Sex
avg_income_by_sex = df[['Sex', 'Income']].groupby('Sex', as_index=False).mean()
print(avg_income_by_sex)

# 7. Settlement Size vs Age
settlement_size_vs_age = df[['Settlement_size', 'Age']].groupby('Settlement_size', as_index=False).mean()
print(settlement_size_vs_age)

# 8. Education vs Marital Status
education_vs_marital_status = df[['Education', 'Marital_status']].groupby('Education', as_index=False).mean()
print(education_vs_marital_status)

# Visualisasi
sns.catplot(data=df, x='Sex', y='Settlement_size', kind='bar', height=4, aspect=1)
plt.title("Sex vs Settlement Size")
plt.show()

sns.catplot(data=df, x='Marital_status', y='Settlement_size', kind='bar', height=4, aspect=1)
plt.title("Marital Status vs Settlement Size")
plt.show()

sns.histplot(df['Age'], kde=True, bins=5, color='blue')
plt.title("Age Distribution")
plt.show()

sns.catplot(data=df, x='Education', y='Income', kind='bar', height=4, aspect=1)
plt.title("Income vs Education")
plt.show()

sns.catplot(data=df, x='Occupation', y='Income', kind='bar', height=4, aspect=1)
plt.title("Occupation vs Income")
plt.show()

# KMeans Clustering
X = df[['Age', 'Income', 'Education', 'Settlement_size', 'Smoker']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X_scaled)

df['Cluster'] = kmeans.labels_

# Visualisasi hasil Clustering
sns.scatterplot(data=df, x='Age', y='Income', hue='Cluster', palette='viridis', s=100)
plt.title('Clustering Results with K-Means')
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()

# Menyimpan DataFrame dengan hasil clustering
df.to_csv('Clustering_with_Clusters.csv', index=False)

# Menyediakan file untuk diunduh
files.download('Clustering_with_Clusters.csv')

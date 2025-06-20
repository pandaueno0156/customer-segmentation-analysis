import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

from sklearn.decomposition import PCA
import warnings

# Suppress the specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')

# Load the data
df = pd.read_csv('customer_data.csv')

# Display the first few rows of the dataset
# print(df.head())

print(df.columns)

# Select relevant columns ((variable 6 to 11))
cluster_raw_df = df[[
    'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
    'MntSweetProducts', 'MntOtherProds'
]]


# Data Summary
print(cluster_raw_df.head())
sns.boxplot(data=cluster_raw_df)
plt.xticks(rotation=90)
plt.title('Boxplot of Customer Data')
plt.tight_layout()
# plt.show()

print(cluster_raw_df.describe())
print(cluster_raw_df.info())

print(f'cluster_raw_df: \n{cluster_raw_df}')

# Correlation matrix to check multicollinearity
# Shows that correlations are below .8 so multicollinearity is no problem. 
# correlation below 0.8 is considered low correlation.

correlation_matrix = cluster_raw_df.corr()
# print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
# plt.show()

# Check for problematic values before clustering
print("Data statistics before scaling:")
print(cluster_raw_df.describe())

# Check for zero or near-zero values
zero_cols = (cluster_raw_df == 0).sum()
print(f"Columns with zero values:\n{zero_cols}")

# Add small constant to zero values if needed
cluster_raw_df_clean = cluster_raw_df.copy()
for col in cluster_raw_df_clean.columns:
    if cluster_raw_df_clean[col].min() == 0:
        cluster_raw_df_clean[col] = cluster_raw_df_clean[col] + 1e-8

# Standardize the cleaned data
scaler = StandardScaler()
cluster_raw_df_scaled = scaler.fit_transform(cluster_raw_df_clean)

# Add small epsilon to prevent division by zero
epsilon = 1e-10
cluster_raw_df_scaled = cluster_raw_df_scaled + epsilon

print(f'cluster_raw_df_scaled: \n{cluster_raw_df_scaled}')
cluster_scaled_df = pd.DataFrame(cluster_raw_df_scaled, columns=cluster_raw_df.columns)
print(f'cluster_scaled_df: \n{cluster_scaled_df}')

# Hiercharical Clustering Analysis
# Calculate distance and linkage
distances_matrix = pdist(cluster_scaled_df, metric='euclidean')
linkage_matrix = linkage(distances_matrix, method='ward')
# print(f'linkage_matrix: \n{linkage_matrix}')


# Plot the Scree Plot
# This helps with elbow method to decide the number of clusters
# We can see that there are elbows at 3 clusters and 4 clusters
heights = linkage_matrix[:, 2]

plt.figure(figsize=(10, 7))
plt.plot(range(10, 0, -1), heights[-10:], marker='o')
plt.title('Scree Plot')
plt.xlabel('Number of Clusters')
plt.ylabel('Height')
# plt.show()

# Plot the dendrogram
# We can see there can be either 3 big clusters or 4 big clusters from the graph
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix, labels=cluster_scaled_df.index, truncate_mode='lastp', p=12)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster size')
plt.ylabel('Distance')
# plt.show()


# K-means Clustering
kmeans_3 = KMeans(n_clusters=3, random_state=42)
kmeans_4 = KMeans(n_clusters=4, random_state=42)

kmeans_3.fit(cluster_scaled_df)
kmeans_4.fit(cluster_scaled_df)

# Cluster size for 3 clusters and 4 clusters
print("K-Means Clustering with 3 clusters: ", np.bincount(kmeans_3.labels_)) # [961 400 276]
print("K-Means Clustering with 4 clusters: ", np.bincount(kmeans_4.labels_)) # [340 225 155 917]

# Cluster means for 3 clusters
cluster_means_3 = cluster_raw_df.groupby(kmeans_3.labels_).mean()
print(f'cluster_means_3: \n{cluster_means_3}')

# Cluster means for 4 clusters
cluster_means_4 = cluster_raw_df.groupby(kmeans_4.labels_).mean()
print(f'cluster_means_4: \n{cluster_means_4}')

# Add cluster labels to df
cluster_raw_df_with_clusters = cluster_raw_df.copy()
cluster_raw_df_with_clusters['cluster_3'] = kmeans_3.labels_
cluster_raw_df_with_clusters['cluster_4'] = kmeans_4.labels_
print(f'cluster_raw_df_with_clusters: \n{cluster_raw_df_with_clusters}')


# Visualize clusters using PCA

# 4 Clusters
pca = PCA(n_components=2)
components = pca.fit_transform(cluster_scaled_df)

plt.figure(figsize=(10, 7))
plt.scatter(components[:, 0], components[:, 1], c=kmeans_4.labels_, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('K-Means 4 Clusters (PCA Projection)')
# plt.show()


# 3 Clusters
pca = PCA(n_components=2)
components = pca.fit_transform(cluster_scaled_df)

plt.figure(figsize=(10, 7))
plt.scatter(components[:, 0], components[:, 1], c=kmeans_3.labels_, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('K-Means 3 Clusters (PCA Projection)')
# plt.show()

# We can see with 3 clusters, clusters are more differentiated
# Therefore, for the analysis, we will go with 3 clusters

cluster_raw_df_with_clusters.drop(columns=['cluster_4'], inplace=True)

# Create a total column to combine all purchase amount
purchase_amount_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntOtherProds']
cluster_raw_df_with_clusters['total_purchase_amount'] = cluster_raw_df_with_clusters[purchase_amount_columns].sum(axis=1)
print(f'cluster_raw_df_with_clusters: \n{cluster_raw_df_with_clusters}')

# Average value by column by cluster group
print(cluster_raw_df_with_clusters.groupby('cluster_3').mean())
# Customer Segmentation Analysis for Supermarket Clients

## Overview
This project provides a **customer segmentation analysis** for a supermarket client using clustering techniques. The goal is to group customers based on their purchasing behavior and recommend targeted marketing strategies for each segment to enhance sales and engagement.

The supermarket provided anonymized customer purchase data across various product categories. Using both **hierarchical clustering** and **K-Means clustering**, we identified key customer groups and provided tailored business insights.

---

## Dataset Description
The dataset `customer_data.csv` includes customer-level purchase data, specifically:

| Column             | Description                                 |
|--------------------|---------------------------------------------|
| MntWines           | Amount spent on wine                        |
| MntFruits          | Amount spent on fruit                       |
| MntMeatProducts    | Amount spent on meat products               |
| MntFishProducts    | Amount spent on fish products               |
| MntSweetProducts   | Amount spent on sweet products              |
| MntOtherProds      | Amount spent on other products              |

Only the six product-related variables were used for segmentation purposes.

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Visualized data distributions using **boxplots**.
- Checked for **outliers** and **zero-value skew**.
- Performed **correlation analysis** to ensure no multicollinearity (all pairwise correlations < 0.8).

### 2. Data Preprocessing
- Replaced zero values with a small constant (ε = 1e-8) to avoid division errors during standardization.
- Scaled the data using **StandardScaler**.

### 3. Clustering
#### Hierarchical Clustering
- Applied **Ward's method** using **Euclidean distance**.
- Generated a **dendrogram** and **scree plot** to estimate optimal cluster count (3–4 clusters).

#### K-Means Clustering
- Tested with **3 and 4 clusters**.
- Applied **PCA** to visualize the clusters in 2D space.

### 4. Cluster Selection
- After evaluating visual separability and interpretability, we chose **3 clusters** for the final analysis.

---

## File Structure
<pre> 
customer-segmentation-analysis/
│
├── customer_data.csv           # Raw dataset
├── segmentation_analysis.ipynb # Python notebook for full analysis
└── README.md                   # Project overview and documentation
<pre> 

---

## Cluster Profiles (K-Means, 3 Clusters)

| Cluster | Segment Name               | Characteristics                                                                 |
|---------|----------------------------|----------------------------------------------------------------------------------|
| 0       | Not-so-Profitable Customers| Lowest spending across all product categories                                   |
| 1       | Most Valuable Customers    | Highest spending across all categories **except wine**                          |
| 2       | Wine Lovers                | Highest spending on **wine**, moderate elsewhere                                |

---

## Business Recommendations

| Segment                     | Recommendation                                                                 |
|-----------------------------|----------------------------------------------------------------------------------|
| Not-so-Profitable Customers| Offer **threshold-based coupons** to incentivize higher spending.               |
| Most Valuable Customers    | Introduce **wine-based recipe content** (e.g., wine + fish meals) to boost wine sales. |
| Wine Lovers                | Cross-sell with **wine-pairing products** like cheese, meats, or recipe bundles. |

---

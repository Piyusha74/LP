import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns

# Load dataset
df = pd.read_csv('sales_data_sample.csv', encoding='unicode_escape')

df.head()

df.info()

# Drop unnecessary columns
to_drop = ['ADDRESSLINE1', 'ADDRESSLINE2', 'STATE', 'POSTALCODE', 'PHONE']
df = df.drop(to_drop, axis=1)

#Check for null values
df.isnull().sum()

df.dtypes

# Select numeric columns only
df_numeric = df.select_dtypes(include=['int64', 'float64'])

# Visualize outliers
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_numeric)
plt.title("Outlier Detection using Boxplots (Numeric Columns Only)")
plt.show()

# Identify outliers using IQR
Q1 = df_numeric.quantile(0.25)
Q3 = df_numeric.quantile(0.75)
IQR = Q3 - Q1

# Count outliers per column
outliers = ((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).sum()
print("\nNumber of Outliers per Numeric Feature:\n", outliers)

# ✅ Select only numeric columns to scale
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df_numeric = df[numeric_cols]

# ✅ Apply StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

print("✅ Data normalized using StandardScaler.")

# ✅ Create DataFrame from scaled data (use original column names)
df_normalized = pd.DataFrame(X_scaled, columns=df_numeric.columns)

# ✅ Display results
print("\nSample of Normalized Data:")
display(df_normalized.head())

print("\nMean of each feature after normalization:\n", df_normalized.mean())
print("\nStandard deviation of each feature after normalization:\n", df_normalized.std())

inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, marker='o')
plt.title("Elbow Method to Determine Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()


from sklearn.metrics import silhouette_score

for k in range (2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)
    print(f"K = {k} → Silhouette Score = {sil_score:.4f}")

# ✅ Run K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# ✅ Add cluster labels to original dataframe
df_combined = df.copy()
df_combined["Cluster"] = labels

# ✅ Plot clusters correctly (use `data=` parameter)
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df_combined,
    x="SALES",        # 🔹 replace with your actual column name
    y="MSRP",         # 🔹 replace with your actual column name
    hue="Cluster",    # 🔹 column name added above
    palette="Set2"
)
plt.title("K-Means Clustering Visualization (K = 3)")
plt.xlabel("Sales")
plt.ylabel("MSRP")
plt.legend(title="Cluster")
plt.show()
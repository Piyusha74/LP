# ==============================
# 📘 Import Libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Suppress warnings
warnings.filterwarnings("ignore")
# ==============================
# 📂 Load Dataset
# ==============================
data = pd.read_csv("uber.csv")
# Create a copy
df = data.copy()
df.head()


# Display info
print("Initial Data Info:")
df.info()

# ==============================
# 🕓 Convert pickup_datetime
# ==============================
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors='coerce')
# Check updated info
print("\nAfter Converting pickup_datetime:")
df.info()

# ==============================
# 📊 Summary Statistics
# ==============================
print("\nSummary Statistics:")
print(df.describe())
# ==============================
# ❓ Missing Values
# ==============================
print("\nMissing Values Before Drop:")
print(df.isnull().sum())
# ==============================
# 💡 Drop Rows with Missing Values
# ==============================
df.dropna(inplace=True)
print("\nMissing Values After Drop:")
print(df.isnull().sum())
# ==============================
# 📦 Outlier Removal (fare_amount)
# ==============================
plt.figure(figsize=(6,4))
plt.boxplot(df['fare_amount'])
plt.title("Boxplot Before Outlier Removal")
plt.show()
# Remove outliers using IQR
q_low = df["fare_amount"].quantile(0.01)
q_hi  = df["fare_amount"].quantile(0.99)
df = df[(df["fare_amount"] >= q_low) & (df["fare_amount"] <= q_hi)]
plt.figure(figsize=(6,4))
plt.boxplot(df['fare_amount'])
plt.title("Boxplot After Outlier Removal")
plt.show()
# ==============================
# 🔢 Feature Preparation
# ==============================
# Convert pickup_datetime to numeric timestamp
df["pickup_datetime"] = pd.to_numeric(df["pickup_datetime"])

# Drop non-numeric columns (if any)
x = df.drop("fare_amount", axis=1)
x = x.select_dtypes(include=[np.number])  # ✅ keep only numeric columns
# Define target variable
y = df["fare_amount"]
# ==============================
# ✂️ Train-Test Split
# ==============================
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1
)
# ==============================
# 📈 Linear Regression
# ==============================
lrmodel = LinearRegression()
lrmodel.fit(x_train, y_train)

predict = lrmodel.predict(x_test)
# Evaluation
lr_rmse = np.sqrt(mean_squared_error(y_test, predict))
lr_r2 = r2_score(y_test, predict)

print("\n✅ Linear Regression Performance:")
print("RMSE:", round(lr_rmse, 3))
print("R²:", round(lr_r2, 3))
# ==============================
# 🌲 Random Forest Regressor (Optimized)
# ==============================
rfrmodel = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,     # ✅ speed improvement
    n_jobs=-1,        # ✅ use all cores
    random_state=101,
    verbose=1
)

print("\nTraining Random Forest Regressor...")
rfrmodel.fit(x_train, y_train)
rfrmodel_pred = rfrmodel.predict(x_test)
rfr_rmse = np.sqrt(mean_squared_error(y_test, rfrmodel_pred))
rfr_r2 = r2_score(y_test, rfrmodel_pred)

print("\n✅ Random Forest Performance:")
print("RMSE:", round(rfr_rmse, 3))
print("R²:", round(rfr_r2, 3))
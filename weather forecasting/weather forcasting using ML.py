#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# Root mean squared error
def root_mean_squared_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

# Load dataset
data = pd.read_csv("C:/Users/hp/Documents/elnino.csv")

# Print column names
print("Column Names:")
print(data.columns)

# Drop rows with missing values
data.dropna(inplace=True)

# Convert non-numeric columns to numeric or drop them
# For example, if the '.' indicates missing values, you can replace them with NaN and then drop those rows:
data.replace('.', pd.NA, inplace=True)
data.dropna(inplace=True)

# Remove leading or trailing whitespaces from column names
data.columns = data.columns.str.strip()

# Define features and target variable
features = ['Year', 'Month', 'Day', 'Latitude', 'Longitude', 'Zonal Winds', 'Meridional Winds', 'Humidity', 'Sea Surface Temp']
target = 'Air Temp'

X = data[features]
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n\n")
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_rmse = root_mean_squared_error(y_test, lr_pred)
print("Linear Regression RMSE:", lr_rmse)

# Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_rmse = root_mean_squared_error(y_test, rf_pred)
print("Random Forest RMSE:", rf_rmse)

# Decision Tree
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_rmse = root_mean_squared_error(y_test, dt_pred)
print("Decision Tree RMSE:", dt_rmse)
print("\n\n")
# Calculate R^2 score for each algorithm
lr_r2 = r2_score(y_test, lr_pred) * 100
rf_r2 = r2_score(y_test, rf_pred) * 100
dt_r2 = r2_score(y_test, dt_pred) * 100
print("\n\n")
# Print R^2 scores as percentages
print("=================Accuracy percentages==================")
print("Linear Regression :", lr_r2, "%")
print("Random Forest :", rf_r2, "%")
print("Decision Tree :", dt_r2, "%")
print("\n\n")
# Tabulation of Predictions Times by Used Algorithm
print("------------------------tabulation of prediction accuracy---------------------------")
predictions_df = pd.DataFrame({
    'Actual': y_test.values,
    'Linear Regression': lr_pred,
    'Random Forest': rf_pred,
    'Decision Tree': dt_pred
})
print(predictions_df.head(10))  # Display the first 10 rows

print("\n\n")

# Visualizations
# Convert y_test to numeric
y_test_numeric = pd.to_numeric(y_test, errors='coerce')

# Drop rows with missing values in y_test and corresponding rows in lr_pred, rf_pred, dt_pred
y_test_numeric = y_test_numeric.dropna()
lr_pred_matched = lr_pred[:len(y_test_numeric)]
rf_pred_matched = rf_pred[:len(y_test_numeric)]
dt_pred_matched = dt_pred[:len(y_test_numeric)]

# Actual vs. Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test_numeric, lr_pred_matched, color='blue', label='Linear Regression')
plt.scatter(y_test_numeric, rf_pred_matched, color='green', label='Random Forest')
plt.scatter(y_test_numeric, dt_pred_matched, color='red', label='Decision Tree')
plt.plot([y_test_numeric.min(), y_test_numeric.max()], [y_test_numeric.min(), y_test_numeric.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted values')
plt.legend()
plt.show()


# RMSE comparison
models = ['Linear Regression', 'Random Forest', 'Decision Tree']
rmses = [lr_rmse, rf_rmse, dt_rmse]
plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=rmses)
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.title('Root Mean Squared Error (RMSE) Comparison')
plt.show()

# Additional Graphical Elements

# Convert y_test to numeric
y_test_numeric = pd.to_numeric(y_test, errors='coerce')

# Drop rows with missing values in y_test and corresponding rows in lr_pred
y_test_numeric = y_test_numeric.dropna()

# Ensure lr_pred and y_test have the same length
lr_pred_matched = lr_pred[:len(y_test_numeric)]


# Residual Plot for Linear Regression
plt.figure(figsize=(10, 6))
sns.residplot(x=lr_pred_matched, y=y_test_numeric - lr_pred_matched, lowess=True, color='blue')
plt.title('Residual Plot - Linear Regression')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.show()



# Convert y_test to numeric
y_test_numeric = pd.to_numeric(y_test, errors='coerce')

# Drop rows with missing values in y_test and corresponding rows in rf_pred
y_test_numeric = y_test_numeric.dropna()
rf_pred_matched = rf_pred[:len(y_test_numeric)]


# Density Plot of Residuals for Random Forest
plt.figure(figsize=(10, 6))
sns.kdeplot(y_test_numeric - rf_pred_matched, shade=True)
plt.title('Density Plot of Residuals - Random Forest')
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.show()

# Prediction Interval Plot for Random Forest
plt.figure(figsize=(10, 6))
plt.scatter(y_test, rf_pred, color='green', label='Predictions')
plt.errorbar(y_test, rf_pred, yerr=1.96 * rf_rmse, fmt='o', color='red', ecolor='lightgray', linestyle='None', label='95% Prediction Interval')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Prediction Interval Plot - Random Forest')
plt.legend()
plt.show()




# Weather Data vs Prediction Time Line Graph
plt.figure(figsize=(10, 6))
plt.hist(y_test.values, bins=20, alpha=0.5, color='black', label='Actual')
plt.hist(dt_pred, bins=20, alpha=0.5, color='red', label='Decision Tree')
plt.xlabel('Air Temp')
plt.ylabel('Frequency')
plt.title('Weather Data vs Prediction Histogram')
plt.legend()
plt.show()

# Replace these placeholder values with actual values from your dataset or provide appropriate values
latitude_value = 40.7128 
longitude_value = -74.0060  
zonal_winds_value = 5.0  
meridional_winds_value = 3.0  
humidity_value = 70.0  
sea_surface_temp_value = 20.0 

# Create dummy data for prediction for the next 10 days
forecast_days = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]  # Assuming forecast_days are consecutive days
dummy_data = pd.DataFrame({
    'Year': [2024] * 10,
    'Month': [3] * 10,
    'Day': forecast_days,
    'Latitude': [latitude_value] * 10,
    'Longitude': [longitude_value] * 10,
    'Zonal Winds': [zonal_winds_value] * 10,
    'Meridional Winds': [meridional_winds_value] * 10,
    'Humidity': [humidity_value] * 10,
    'Sea Surface Temp': [sea_surface_temp_value] * 10
})

# Use the trained Random Forest model to predict air temperature for the next 10 days
predictions = rf_model.predict(dummy_data)

# Display the predictions for the next 10 days
print("Predicted Air Temperature for the Next 10 Days:")
for day, temp in zip(forecast_days, predictions):
    print(f"Day {day}: {temp}°C")
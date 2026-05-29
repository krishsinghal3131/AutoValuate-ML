import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics

# =====================================================================
# 1. DATA LOADING & PREPROCESSING
# =====================================================================

# Load the dataset
car_dataset = pd.read_csv('car data.csv')

# Encoding categorical columns to numerical values
car_dataset.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
car_dataset.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)
car_dataset.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)

# Splitting Features (X) and Target Variable (Y)
# Dropping Car_Name (not a numerical feature) and Selling_Price (target)
X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
Y = car_dataset['Selling_Price']

# Splitting the data into Training and Test sets (80% Train, 20% Test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)


# =====================================================================
# 2. LINEAR REGRESSION MODEL
# =====================================================================

# Loading and training the Linear Regression model
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)

# Model Evaluation on Training Data
training_data_prediction = lin_reg_model.predict(X_train)
error_score_train = metrics.r2_score(Y_train, training_data_prediction)
print(f"Linear Regression - Training R-squared Error: {error_score_train}")

# Model Evaluation on Test Data
test_data_prediction = lin_reg_model.predict(X_test)
error_score_test = metrics.r2_score(Y_test, test_data_prediction)
print(f"Linear Regression - Test R-squared Error: {error_score_test}\n")


# =====================================================================
# 3. LASSO REGRESSION MODEL
# =====================================================================

# Loading and training the Lasso Regression model
lasso_reg_model = Lasso()
lasso_reg_model.fit(X_train, Y_train)

# Model Evaluation on Training Data
lasso_train_prediction = lasso_reg_model.predict(X_train)
error_score_lasso_train = metrics.r2_score(Y_train, lasso_train_prediction)
print(f"Lasso Regression - Training R-squared Error: {error_score_lasso_train}")

# Model Evaluation on Test Data
lasso_test_prediction = lasso_reg_model.predict(X_test)
error_score_lasso_test = metrics.r2_score(Y_test, lasso_test_prediction)
print(f"Lasso Regression - Test R-squared Error: {error_score_lasso_test}\n")


# =====================================================================
# 4. VISUALIZATION (ACTUAL VS PREDICTED PRICES)
# =====================================================================

# Plotting the results for Linear Regression Test Predictions
plt.scatter(Y_test, test_data_prediction, color='blue', label='Linear Regression')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.legend()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the data
train_df = pd.read_csv('train.csv')

# Select the relevant features
X = train_df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = train_df['SalePrice']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model using mean squared error, R-squared, and mean absolute error
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')
print(f'Mean Absolute Error: {mae:.2f}')

# Plot the data
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=X_train['GrLivArea'], y=y_train)
plt.title('GrLivArea vs SalePrice (Training Data)')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')

plt.subplot(1, 2, 2)
sns.scatterplot(x=X_val['GrLivArea'], y=y_val)
plt.title('GrLivArea vs SalePrice (Validation Data)')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')

plt.tight_layout()
plt.show()

# Plot the predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred)
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Predicted vs Actual SalePrice')
plt.show()

# Plot the residuals
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, y_pred - y_val)
plt.xlabel('Predicted SalePrice')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted SalePrice')
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load the house price dataset
data = pd.read_csv("train.csv")


# Display the first five rows to inspect the dataset
print(data.head())


# Select the input features used for prediction
# GrLivArea     -> Above-ground living area in square feet
# BedroomAbvGr  -> Number of bedrooms above ground
# FullBath      -> Number of full bathrooms
features = ["GrLivArea", "BedroomAbvGr", "FullBath"]

# Define the target variable
# SalePrice -> Sale price of the house
target = "SalePrice"


# Separate the input features (X) and target variable (y)
X = data[features]
y = data[target]


# Split the dataset into training and testing sets
# 80% of the data is used for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create a Linear Regression model
model = LinearRegression()


# Train the model using the training data
model.fit(X_train, y_train)


# Use the trained model to predict house prices
# for the test dataset
y_pred = model.predict(X_test)


# Calculate the Mean Squared Error (MSE)
# Lower MSE indicates smaller prediction errors
mse = mean_squared_error(y_test, y_pred)


# Calculate the R-squared (R²) score
# A value closer to 1 indicates better model performance
r2 = r2_score(y_test, y_pred)


# Display the model evaluation results
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.4f}")


# Create sample data for a new house
# Living area: 2000 sq. ft.
# Bedrooms: 3
# Full bathrooms: 2
new_house = pd.DataFrame(
    [[2000, 3, 2]],
    columns=features
)


# Predict the selling price of the new house
predicted_price = model.predict(new_house)[0]


# Display the predicted house price
print(f"Predicted House Price: ${predicted_price:,.2f}")


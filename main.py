import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Dataset: Features (Square Meters) and Labels (Price in USD)
# x = Independent variable, y = Dependent variable
data = {
    'sq_meters': [50, 70, 80, 100, 120, 150, 200, 250],
    'price_usd': [150000, 200000, 240000, 300000, 350000, 450000, 600000, 750000]
}

df = pd.DataFrame(data)

# 2. Reshaping data for Scikit-Learn (Needs 2D array for X)
X = df[['sq_meters']] 
y = df['price_usd']

# 3. Model Initialization and Training
model = LinearRegression()
model.fit(X, y) # Training the 'brain'

# 4. Making a Prediction
test_value = [[135]] # Predictive ask: How much for 135 sq meters?
prediction = model.predict(test_value)

print(f"Prediction for {test_value[0][0]} sqm: ${prediction[0]:.2f}")

# 5. Visualizing the results (Optional but great for GitHub)
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.title('House Price Prediction - Day 1')
plt.xlabel('Square Meters')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
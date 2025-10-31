
# LINEAR REGRESSION MODEL FOR ACCIDENT SEVERITY

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Mock Dataset
data = {
    'Speed': np.random.randint(40, 140, 100),               # Vehicle speed (km/h)
    'Road_Surface': np.random.choice([0, 1], 100),          # 0 = Dry, 1 = Wet
    'Weather': np.random.choice([0, 1, 2], 100),            # 0 = Clear, 1 = Rainy, 2 = Foggy
    'Driver_Age': np.random.randint(18, 70, 100),           # Age of driver
    'Time_of_Day': np.random.choice([0, 1], 100),           # 0 = Day, 1 = Night
    'Seatbelt': np.random.choice([0, 1], 100),              # 1 = Wearing seatbelt, 0 = Not wearing
}

df = pd.DataFrame(data)

# Dependent Variable (Accident Severity)
# Formula for synthetic data (for realism)
df['Accident_Severity'] = (
    2
    + 0.05 * df['Speed']
    + 1.2 * df['Road_Surface']
    + 0.8 * df['Weather']
    - 0.02 * df['Driver_Age']
    + 0.6 * df['Time_of_Day']
    - 1.5 * df['Seatbelt']
    + np.random.normal(0, 0.5, 100)  # Random noise
)

# Define Features (X) and Target (y)
X = df[['Speed', 'Road_Surface', 'Weather', 'Driver_Age', 'Time_of_Day', 'Seatbelt']]
y = df['Accident_Severity']

# Split into Train/Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate Model Performance
y_pred = model.predict(X_test)
print("Model Evaluation Metrics:")
print("--------------------------")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# Coefficients (to interpret impact)
coefficients = pd.DataFrame({
    'Variable': X.columns,
    'Coefficient': model.coef_
})
print("\nFeature Importance:")
print(coefficients)

#Predict Severity for a Hypothetical Case
# Example input:
# Speed = 100 km/h, Wet road, Rainy weather, Driver age 35, Nighttime, No seatbelt
sample_data = pd.DataFrame({
    'Speed': [100],
    'Road_Surface': [1],
    'Weather': [1],
    'Driver_Age': [35],
    'Time_of_Day': [1],
    'Seatbelt': [0]
})

predicted_severity = model.predict(sample_data)[0]
print("\nPredicted Accident Severity for Sample Case:", round(predicted_severity, 2))

#Interpretation
print("\nInterpretation:")
print("The model predicts a severity score of around", round(predicted_severity, 2))
print("Higher values indicate more severe accidents.")
print("Speed, wet roads, poor weather, and lack of seatbelt use all increase severity risk.")

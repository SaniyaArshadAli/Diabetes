import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Set up the Streamlit app
st.title("Diabetes Progression Prediction using Random Forest")

# Load the dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target, name="Disease Progression")

st.subheader("Dataset Preview")
st.write(X.head())

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar for model parameters
st.sidebar.header("Model Parameters")
n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 10, 200, 100, step=10)
max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)

# Train the model
model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Show evaluation metrics
st.subheader("Model Evaluation")
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# Prediction on user input
st.subheader("Try Prediction Yourself")

input_data = {}
for feature in diabetes.feature_names:
    input_data[feature] = st.number_input(f"{feature}", value=float(X[feature].mean()))

if st.button("Predict Disease Progression"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Disease Progression Value: {prediction:.2f}")

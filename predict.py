# predict_aluminium.py
import pandas as pd
import joblib
import numpy as np

# Load the saved model and scaler
pls = joblib.load('pls_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load the test data
data_test_path = r"C:\Users\kaout\OneDrive\Documents\S2\Capteurs&actionneurs\projet\book8.xlsx"
data_test = pd.read_excel(data_test_path)
data_test.columns = data_test.columns.astype(str)

# Normalize the test features using the same scaler as in training
X_test_scaled = scaler.transform(data_test)

# Predict Mg concentration
y_pred = pls.predict(X_test_scaled)
predicted_mg = y_pred.flatten()[0]

# Define the aluminium types
mg_concentration_dict = {
    '3004': 1.26, '3005': 0.45, '3105': 0.5, '5454': 2.66, '6061': 0.985,
    '6111': 0.77, '6351': 0.61, '413': 0.0001
}

# Find the aluminium type
def find_aluminium_type(predicted_mg, mg_dict, tolerance=0.04):
    for key, value in mg_dict.items():
        if abs(value - predicted_mg) < tolerance:
            return key
    return "Inconnu"

predicted_type = find_aluminium_type(predicted_mg, mg_concentration_dict, tolerance=0.05)

# Print the predicted Mg concentration and the aluminium type
print(f"Concentration prédite de Mg : {predicted_mg}")
print(f"Type d'aluminium prédit: {predicted_type}")

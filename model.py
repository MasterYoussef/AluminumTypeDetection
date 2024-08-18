# save_model.py
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

# Load and preprocess the training data
data_al_spectre_path = r"C:\Users\kaout\OneDrive\Documents\S2\Capteurs&actionneurs\projet\Al_Spectre_tr.xlsx"
data_al_spectre = pd.read_excel(data_al_spectre_path)
data_al_spectre.columns = data_al_spectre.columns.astype(str)

# Prepare training features and target
X = data_al_spectre.iloc[:, 1:].drop(columns=['Mg'])
y = data_al_spectre['Mg']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train PLS Regression Model
pls = PLSRegression(n_components=20)
pls.fit(X_train, y_train)

# Predict and evaluate on the test set
y_pred = pls.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# Save the model and the scaler
joblib.dump(pls, 'pls_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

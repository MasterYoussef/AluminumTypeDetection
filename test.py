import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import StandardScaler
pd.set_option('display.float_format', '{:.4f}'.format)
data_al_spectre_path = r"C:\Users\kaout\OneDrive\Documents\S2\Capteurs&actionneurs\projet\Al_Spectre_tr.xlsx"

data_al_spectre = pd.read_excel(data_al_spectre_path)

data_al_spectre.columns = data_al_spectre.columns.astype(str)

X = data_al_spectre.drop(columns=['Mg'])
y = data_al_spectre['Mg']
print(X.shape)
print(y.shape)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Configuration du modèle PLS
pls = PLSRegression(n_components=20) 
# Entraînement du modèle
pls.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = pls.predict(X_test)
# Évaluation du modèle
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R²: {r2}")
# Création du graphique
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Real')  # Points de prédiction
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit')  # Ligne parfaite
plt.text(x=0.05, y=0.8, s=f'RMSE: {rmse:.3f}\nR²: {r2:.3f}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', alpha=0.1))

plt.xlabel('Real Mg Concentration')
plt.ylabel('Predicted Mg Concentration')
plt.title('Comparison of Real and Predicted Mg Concentrations')
plt.legend()
plt.grid(True)
plt.show()



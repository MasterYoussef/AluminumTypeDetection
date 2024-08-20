# Magnesium Concentration Prediction using Lasso Regression

ThThis project uses machine learning algorithms, including PLS, LASSO, and Linear Regression, to classify aluminum types based on spectroscopy sensor data. The models are evaluated using R² scores

## Project Overview

The goal of this project is to develop a model that can accurately predict the Magnesium concentration in aluminum alloys using spectral data. We use Lasso Regression, a linear regression technique with L1 regularization, which helps in feature selection and preventing overfitting.

## Dependencies

To run this project, you'll need the following Python libraries:

- numpy
- pandas
- scikit-learn
- matplotlib

You can install these dependencies using pip:

```
pip install numpy pandas scikit-learn matplotlib
```

## Data

The project uses spectral data stored in an Excel file named `Al_Spectre_tr.xlsx`. This file should be placed in the appropriate directory (currently set to `C:\Users\Asus\Desktop\Projet_ALUM\`).

## Usage

1. Load and preprocess the data:
   - The script reads the Excel file and prepares the features (X) and target variable (y).

2. Split the data into training and testing sets:
   - We use an 80-20 split for training and testing.

3. Train the Lasso Regression model:
   - The model is trained using the training data.

4. Evaluate the model:
   - We calculate the Root Mean Square Error (RMSE) and R-squared (R²) score for both training and test sets.

5. Visualize the results:
   - A scatter plot is created to compare predicted vs. actual Mg concentrations.

## Results

The current model achieves the following performance:

- Training RMSE: 0.0351
- Training R²: 0.9965
- Test RMSE: 0.0723
- Test R²: 0.9950

These results indicate that the model performs well in predicting Mg concentrations, with high R² values suggesting a good fit.

## Future Improvements

- Try other regression techniques like Ridge Regression or Elastic Net for comparison.
- Collect more data or engineer new features to potentially improve prediction accuracy.

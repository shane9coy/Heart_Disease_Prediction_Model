import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import os
from sklearn.model_selection import train_test_split, GridSearchCV # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score # type: ignore
import joblib # type: ignore


# Data reference 
FEATURE_RANGES = {
    'age': (20, 100),           # Age in years
    'sex': (0, 1),              # 0 = female, 1 = male
    'cp': (1, 4),               # Chest pain type (1-4)
    'trestbps': (80, 200),      # Resting blood pressure (mm Hg)
    'chol': (100, 600),         # Serum cholesterol (mg/dl)
    'fbs': (0, 1),              # Fasting blood sugar > 120 mg/dl (0 = false, 1 = true)
    'restecg': (0, 2),          # Resting ECG results (0-2)
    'thalach': (60, 220),       # Maximum heart rate achieved
    'exang': (0, 1),            # Exercise-induced angina (0 = no, 1 = yes)
    'oldpeak': (0, 6.2),        # ST depression induced by exercise
    'slope': (1, 3),            # Slope of the peak exercise ST segment (1-3)
    'ca': (0, 3),               # Number of major vessels (0-3)
    'thal': (3, 7)              # Thalium stress test result (3 = normal, 6 = fixed defect, 7 = reversible defect)
}

# Create a function to transmute raw data and save as CSV file
def transmute_data():
    # Load the UCI Heart Disease Dataset & define columns
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
               'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    try:
        data = pd.read_csv(url, header=None, names=columns)
    except Exception as e:
        print(f"Error loading data from URL: {e}")
        return None
    data = data.replace('?', np.nan)  # Replace '?' with NaN

    # Quick look at the data3
    print("\nStep 1: Initial look at the data:")
    print("Data Info:")
    data.info()
    print(f"\nData Shape: {data.shape}")
    print("\nFirst 10 rows of the dataset:")
    print(data.head(10))

    # Identify missing values in the dataset
    print("\nIdentify missing values in the dataset")
    print("Columns with missing values:")
    print(data.isna().sum())

    # Inspect the columns with missing values
    print("\nUnique values in the loaded DataFrame:")
    print(data.thal.unique())
    print(data.ca.unique())

    # Convert 'ca' and 'thal' columns to floats
    data['ca'] = pd.to_numeric(data['ca'], errors='coerce')
    data['thal'] = pd.to_numeric(data['thal'], errors='coerce')
  
    # Fill missing value with  mode implementation
    for col in columns:
        if data[col].isna().sum() > 0:
            # Fill missing values with the mode
            mode_value = data[col].mode()[0]
            data[col] = data[col].fillna(mode_value)
            print(f"\nFilled missing values in '{col}' with mode: {mode_value}")
            data[col] = data[col].astype(int)  # Ensure integer type for categorical columns

    print("\nCheck for updated column changes:")
    print(data.isna().sum())

    # Convert target column to binary values
    print('\nCheck for unique values in target column and convert to binary:')
    print(data['target'].unique())
    data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)

    # Inspect dataset to see if changes were successful
    print('\nCheck target column again for successful conversion to binary values:')
    print(data['target'].unique())
    print("\nUpdated Data Info:")
    data.info()

    # Show the counts of 0 and 1 after transformation
    print("\nCounts of 0 and 1 in target column after transformation:")
    counts = data['target'].value_counts()
    print("No heart disease (0):", counts.get(0, 0))
    print("Heart disease (1):", counts.get(1, 0))

    return data


# Run function, save to CSV, and load CSV into new DataFrame
data = transmute_data()
csv_file = "new_data.csv"  # Changed to match load file

# Add Patient ID column with sequential numbers
data['Patient ID'] = range(1, len(data) + 1)

# Reorder columns to put Patient ID first
columns = ['Patient ID'] + [col for col in data.columns if col != 'Patient ID']
data = data[columns]

# Save to CSV with the new column order
data.to_csv('new_data.csv', index=False)
print(f"\nDataset saved as CSV: {os.path.abspath(csv_file)}")

# Load the CSV file into a new DataFrame for offline work
new_data = pd.read_csv('new_data.csv')  # Original file name

# Verify the loaded DataFrame
print("\nFirst 10 rows of the loaded DataFrame:")
print(new_data.head(10))
print("\nLoaded DataFrame Info:")
new_data.info()

# Calculate and display percentages
total = len(new_data)
counts = new_data['target'].value_counts()
negative_percentage = counts.get(0, 0) / total * 100
positive_percentage = counts.get(1, 0) / total * 100

print("\nPercentage of each class:")
print(f"Total records: {total}")
print(f"No heart disease (0): {negative_percentage:.2f}%")
print(f"Heart disease (1): {positive_percentage:.2f}%")

# Create X & y variables and separate into training (80%) and testing (20%) sets
X = new_data.drop('target', axis=1)
y = new_data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the resulting sets
print("\nShapes of the resulting sets:")
print("Features (X) shape:", X.shape)
print("Target (y) shape:", y.shape)
print("Training set shapes - X_train:", X_train.shape, "y_train:", y_train.shape)
print("Test set shapes - X_test:", X_test.shape, "y_test:", y_test.shape)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data & transform
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Verify the scaling
print("\nMean of X_train_scaled (should be close to 0):")
print(np.mean(X_train_scaled, axis=0))
print("\nMean of X_test_scaled (not necessarily 0, but similar scale):")
print(np.mean(X_test_scaled, axis=0))
print("\nStandard deviation of X_train_scaled (should be close to 1):")
print(np.std(X_train_scaled, axis=0))
print("\n")

# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],  # Number of trees
    'max_depth': [None, 10, 20, 30],            # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],            # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],              # Minimum samples required to be at a leaf node
    'bootstrap': [True, False]                  # Whether bootstrap samples are used
}

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                           cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

# Fit the grid search to the training data
print("\nStarting Grid Search for Hyperparameter Optimization...")
grid_search.fit(X_train_scaled, y_train)

# Get the best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("\nBest Hyperparameters:")
print(best_params)
print(f"Best Cross-Validation Accuracy: {best_score:.4f}")

# Use the best model for predictions
y_pred = best_model.predict(X_test_scaled)

# Compute full evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Print the results
print("\nFull Model Performance on Test Set (Optimized Model):")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"ROC-AUC: {roc_auc:.2f}")

# Feature importance analysis
feature_names = X.columns
importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='c')
plt.xlabel('Importance')
plt.title('Feature Importance in Random Forest Model')
plt.gca().invert_yaxis()  # Highest importance at top
plt.savefig('feature_importance.png')
print("\nFeature importance plot saved as 'feature_importance.png'")
plt.show()

# Save the model, scaler, and feature list using joblib
# Save the optimized model
try:
    model_filename = 'optimized_heart_disease_model.pkl'
    joblib.dump(best_model, model_filename)
    print(f"\nOptimized model saved as '{model_filename}'")
except Exception as e:
    print(f"Error saving model: {e}")

# Save the scaler
try:
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved as 'scaler.pkl'")
except Exception as e:
    print(f"Error saving scaler: {e}")

feature_names = list(X.columns)
feature_names_filename = 'feature_names.pkl'
try:
    joblib.dump(feature_names, feature_names_filename)
    print(f"Feature names saved as '{feature_names_filename}'")
except Exception as e:
    print(f"Error saving feature names: {e}")


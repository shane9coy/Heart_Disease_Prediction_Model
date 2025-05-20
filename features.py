#Constants for feature variables in our  heart disease prediction model

# Features/Column names for the heart disease dataset
FEATURE_NAMES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
                 'ca', 'thal'
]

# Define expected ranges for each feature (for validation)
# These are approximate ranges based on the UCI Heart Disease Dataset
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


FEATURE_DESCRIPTIONS = [
    'Age of the patient in years',
    'Sex of the patient (0 = female, 1 = male)',
    'Chest pain type (1-4)',
    'Resting blood pressure (in mm Hg)',
    'Serum cholesterol (in mg/dl)',
    'Fasting blood sugar > 120 mg/dl (0 = false, 1 = true)',
    'Resting electrocardiographic results (0-2)',
    'Maximum heart rate achieved',
    'Exercise-induced angina (0 = no, 1 = yes)',
    'ST depression induced by exercise relative to rest',
    'Slope of the peak exercise ST segment (1-3)',
    'Number of major vessels (0-3) colored by fluoroscopy',
    'Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)'
]
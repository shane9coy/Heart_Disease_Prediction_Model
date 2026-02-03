# Heart Disease Prediction Model

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning project that predicts heart disease using clinical features from the UCI Cleveland Heart Disease Dataset. The model is built with a Random Forest Classifier and optimized using Scikit-learn + GridSearchCV for hyperparameter tuning.

## Features

This model uses **13 clinical features** to predict heart disease:

| Feature | Description | Range |
|---------|-------------|-------|
| `age` | Age of the patient in years | 20-100 |
| `sex` | Sex of the patient (0 = female, 1 = male) | 0-1 |
| `cp` | Chest pain type (1-4) | 1-4 |
| `trestbps` | Resting blood pressure (in mm Hg) | 80-200 |
| `chol` | Serum cholesterol (in mg/dl) | 100-600 |
| `fbs` | Fasting blood sugar > 120 mg/dl (0 = false, 1 = true) | 0-1 |
| `restecg` | Resting electrocardiographic results | 0-2 |
| `thalach` | Maximum heart rate achieved | 60-220 |
| `exang` | Exercise-induced angina (0 = no, 1 = yes) | 0-1 |
| `oldpeak` | ST depression induced by exercise relative to rest | 0-6.2 |
| `slope` | Slope of the peak exercise ST segment | 1-3 |
| `ca` | Number of major vessels colored by fluoroscopy | 0-3 |
| `thal` | Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect) | 3-7 |

## Dataset

The model uses the **UCI Cleveland Heart Disease Dataset** from the UCI Machine Learning Repository.

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data)
- **Records**: 303 patient records
- **Target Variable**: Binary classification (0 = no heart disease, 1 = heart disease)

### Class Distribution
- No heart disease (0): ~54%
- Heart disease (1): ~46%

## Model

### Algorithm
- **Random Forest Classifier**: An ensemble learning method that operates by constructing a multitude of decision trees at training time.

### Hyperparameter Optimization
- **Method**: GridSearchCV with 5-fold cross-validation
- **Parameters Optimized**:
  - `n_estimators`: [100, 200, 300, 400, 500]
  - `max_depth`: [None, 10, 20, 30]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]
  - `bootstrap`: [True, False]

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Dependencies

Install the required packages:

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

Or install all dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the main training script to:
1. Download and preprocess the dataset
2. Train and optimize the Random Forest model
3. Generate feature importance visualization
4. Save the model, scaler, and feature names

```bash
python main.py
```

### Making Predictions

Use the prediction CLI tool to input patient data and get heart disease predictions:

```bash
python heart_disease_predictor.py
```

The tool will prompt for each clinical feature with validation against expected ranges. Enter values or type 'exit' to quit.

Example session:
```
Heart Disease Prediction Tool
----------------------------
Loaded feature names: ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
Number of features: 13

Enter the patient's data (or type 'exit' to quit):
Age of the patient in years (expected range: (20, 100)): 55
Sex of the patient (0 = female, 1 = male) (expected range: (0, 1)): 1
...
```

## Project Structure

```
Heart disease prediction model/
├── main.py                          # Main training script
├── heart_disease_predictor.py       # Prediction CLI tool
├── features.py                      # Feature definitions and ranges
├── readme.md                        # This file
├── .gitignore                       # Git ignore rules
├── .vscode/                         # VSCode settings
├── 9000/                            # Additional resources
├── data.json                        # Project data
├── new_data.csv                     # Processed dataset
├── prediction_log.csv               # Prediction history log
├── feature_importance.png           # Feature importance visualization
├── feature_names.pkl                # Feature names pickle
├── scaler.pkl                       # StandardScaler pickle
├── optimized_heart_disease_model.pkl # Trained model pickle
└── heart_disease_model.pkl          # Original model pickle
```

## Output Files

| File | Description |
|------|-------------|
| `optimized_heart_disease_model.pkl` | Trained Random Forest model with optimized hyperparameters |
| `scaler.pkl` | StandardScaler fitted on training data for feature normalization |
| `feature_names.pkl` | List of feature names used by the model |
| `feature_importance.png` | Bar chart showing feature importance rankings |
| `prediction_log.csv` | CSV file logging all predictions with timestamps, patient IDs, input values, predictions, and probabilities |

## Data Pipeline

### Preprocessing Steps

1. **Data Loading**: Download dataset from UCI repository
2. **Missing Value Handling**: Replace '?' with NaN, then fill with mode
3. **Type Conversion**: Convert 'ca' and 'thal' columns to numeric types
4. **Target Binarization**: Convert target values to binary (0 or 1)
   - Original values > 0 become 1 (heart disease present)
   - Original values = 0 become 0 (no heart disease)
5. **Feature Scaling**: StandardScaler normalization (mean=0, std=1)
6. **Train-Test Split**: 80% training, 20% testing (random_state=42)

### Pipeline Diagram

```
UCI Dataset → Missing Value Handling → Type Conversion → Target Binarization
                                                                    ↓
Feature Scaling → Train-Test Split → GridSearchCV → Optimized Model
```

## Evaluation Metrics

The model is evaluated on the test set using the following metrics:

| Metric | Description | Target |
|--------|-------------|--------|
| **Accuracy** | Proportion of correct predictions | > 80% |
| **Precision** | True positives / (True positives + False positives) | > 80% |
| **Recall** | True positives / (True positives + False negatives) | > 80% |
| **ROC-AUC** | Area under the ROC curve | > 85% |

### Sample Performance (Test Set)
```
Accuracy:  0.85
Precision: 0.84
Recall:    0.86
ROC-AUC:   0.92
```

## Feature Importance

The model identifies the most important features for heart disease prediction. Common top features include:

1. `ca` - Number of major vessels
2. `thal` - Thalassemia test result
3. `oldpeak` - ST depression
4. `thalach` - Maximum heart rate
5. `cp` - Chest pain type

A feature importance visualization is saved as `feature_importance.png` after training.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) for the Heart Disease Dataset
- [scikit-learn](https://scikit-learn.org/) for the machine learning tools

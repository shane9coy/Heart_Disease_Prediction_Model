import joblib #type:ignore
import pandas as pd #type:ignore
from features import FEATURE_RANGES, FEATURE_DESCRIPTIONS

# Load the saved model and scaler
model = joblib.load('optimized_heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')
patient_id = ""

# Function to get & validate user input
def get_patient_data(feature_names, feature_ranges, feature_descriptions):
    # Validate inputs
    if len(feature_names) != len(feature_ranges) or len(feature_names) != len(feature_descriptions):
        raise ValueError("Mismatch in feature_names, feature_ranges, or feature_descriptions lengths")
    
    print(f"Feature Key & expected ranges: {feature_ranges}")
    while True:
        try:
            patient_id = int(input("\nEnter the patient's ID: "))
            break
        except ValueError:
            print("Please enter a valid patient ID (numbers only)")
    print("\nEnter the patient's data (or type 'exit' to quit):")
    patient_data = {}
    
    for i, feature in enumerate(feature_names):
        while True:  # Loop until valid input is received for this feature
            user_input = input(f"{feature_descriptions[i]} (expected range: {feature_ranges[feature]}): ").strip().lower()
            
            # Allow the user to exit
            if user_input == 'exit':
                print("Exiting data input.")
                return None, None
            
            try:
                value = float(user_input)
                # Validate the input against the expected range
                min_val, max_val = feature_ranges[feature]
                if min_val <= value <= max_val:
                    patient_data[feature] = value
                    break  # Move to the next feature
                else:
                    print(f"Value out of range! Please enter a value between {min_val} and {max_val}.")
            except ValueError:
                print("Invalid input! Please enter a valid number or 'exit' to quit.")
    
    return patient_id, patient_data

# Function to make a prediction
def predict_heart_disease(patient_data, model, scaler, feature_names):
    try:
        # Convert the patient data to a DataFrame
        patient_df = pd.DataFrame([patient_data], columns=feature_names)
        
        # Scale the data
        patient_scaled = scaler.transform(patient_df)
        
        # Make a prediction
        prediction = model.predict(patient_scaled)[0]
        probability = model.predict_proba(patient_scaled)[0][1]  # Probability of class 1 (heart disease)
        
        return prediction, probability
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

# Function to log predictions
def log_prediction(patient_data, prediction, probability, feature_names, patient_id):
    import csv
    import datetime
    
    with open('prediction_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        # Write header if file is empty
        if f.tell() == 0:
            writer.writerow(['Timestamp', 'Patient_ID'] + feature_names + ['Prediction', 'Probability'])
        # Write the prediction data
        row = [datetime.datetime.now(), patient_id] + [patient_data[feat] for feat in feature_names] + [prediction, probability]
        writer.writerow(row)

# Main loop to allow multiple predictions
def main():
    print("Heart Disease Prediction Tool")
    print("----------------------------")
    print("Loaded feature names:", feature_names)
    print("Number of features:", len(feature_names))
    
    while True:
        # Get patient data
        patient_id, patient_data = get_patient_data(feature_names, FEATURE_RANGES, FEATURE_DESCRIPTIONS)
        
        # Check if user exited
        if patient_data is None:
            print("Program terminated by user.")
            break
        
        # Make a prediction
        prediction, probability = predict_heart_disease(patient_data, model, scaler, feature_names)
        
        if prediction is not None:
            # Interpret the prediction
            print(f"\nResults for Patient #{patient_id}")
            if prediction == 1:
                print(f"Prediction: Heart disease likely (Probability of heart disease: {probability:.2f})")
            else:
                print(f"Prediction: No heart disease (Probability of heart disease: {probability:.2f})")
            
            # Log the prediction
            log_prediction(patient_data, prediction, probability, feature_names, patient_id)
        
        # Ask if the user wants to continue
        continue_choice = input("\nWould you like to enter another patient's data? (yes/no): ").strip().lower()
        if continue_choice != 'yes' or 'y':
            print("Exiting the program.")
            break

main()
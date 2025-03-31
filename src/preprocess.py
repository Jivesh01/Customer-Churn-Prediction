import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def load_and_preprocess_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)

    # Convert 'TotalCharges' to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                        'PaperlessBilling', 'PaymentMethod', 'Churn']

    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # Scale numerical features
    numerical_cols = ['MonthlyCharges', 'TotalCharges', 'tenure']
    if all(col in df.columns for col in numerical_cols):
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    else:
        raise ValueError(f"Missing one or more numerical columns: {numerical_cols}")

    # Ensure the models directory exists
    models_dir = "../models"
    os.makedirs(models_dir, exist_ok=True)

    # Save encoders and scalers
    joblib.dump(label_encoders, os.path.join(models_dir, "label_encoders.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

    return df

# Run preprocessing
input_filepath = 'D:/vs_code/Customer_Churn_Prediction/data/Customer_churn.csv'
output_filepath = 'D:/vs_code/Customer_Churn_Prediction/data/Customer_churn_processed.csv'

df = load_and_preprocess_data(input_filepath)
df.to_csv(output_filepath, index=False)
print("Data preprocessing complete.")

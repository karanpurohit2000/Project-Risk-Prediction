# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

print("Starting model training script...")

# --- 1. Load Data ---
# Ensure 'project_risk_data.csv' is in the same directory as this script
try:
    df = pd.read_csv('project_risk_data.csv')
    print("Data loaded successfully from 'project_risk_data.csv'")
    print(f"Dataset shape: {df.shape}")
    print("First 5 rows of data:")
    print(df.head())
except FileNotFoundError:
    print("Error: 'project_risk_data.csv' not found.")
    print("Please ensure you have run the data generation script and the CSV file is in the correct directory.")
    exit() # Exit if data is not found

# --- 2. Data Cleaning and Preprocessing ---

# Drop identifier columns not used for training
# 'project_id' is just an identifier and should not be used as a feature.
# 'start_date', 'due_date', 'actual_end_date' are used to derive 'project_delayed'
# but not directly as features for the model in this simplified example.
# In a real scenario, features like 'days_remaining_until_due' could be engineered.
features_to_drop = ['project_id', 'start_date', 'due_date', 'actual_end_date']
df_processed = df.drop(columns=features_to_drop, errors='ignore')

print(f"\nDropped columns: {features_to_drop}")
print(f"Processed data shape: {df_processed.shape}")

# Check for missing values (synthetic data should be clean, but good practice)
if df_processed.isnull().sum().sum() > 0:
    print("\nWarning: Missing values found. Imputing with median.")
    for col in df_processed.columns:
        if df_processed[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df_processed[col]):
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            else: # For categorical, fill with mode or a placeholder
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])

# Define features (X) and target (y)
# The target variable is 'project_delayed'
X = df_processed.drop('project_delayed', axis=1)
y = df_processed['project_delayed']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print("Features used for training:")
print(X.columns.tolist())

# Split data into training and testing sets
# Using a stratify split to maintain the proportion of delayed/not delayed projects
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set shape (X_train, y_train): {X_train.shape}, {y_train.shape}")
print(f"Testing set shape (X_test, y_test): {X_test.shape}, {y_test.shape}")

# Scale numerical features
# It's crucial to fit the scaler ONLY on the training data to prevent data leakage.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Transform test data using the *fitted* scaler

# Convert back to DataFrame for better readability (optional, but useful for inspection)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

print("\nFeatures scaled successfully.")

# --- 3. Model Selection and Training ---
# Using RandomForestClassifier for its robustness and good performance
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # 'balanced' helps with imbalanced classes
print(f"\nTraining {type(model).__name__} model...")
model.fit(X_train_scaled, y_train)
print("Model training complete.")

# --- 4. Evaluation ---
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] # Probability of being delayed

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Delayed (0)', 'Delayed (1)']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- 5. Save Model and Scaler ---
# Create a 'models' directory if it doesn't exist
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, 'random_forest_model.joblib')
scaler_path = os.path.join(models_dir, 'scaler.joblib')
feature_names_path = os.path.join(models_dir, 'feature_names.joblib')

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(X.columns.tolist(), feature_names_path) # Save feature names for consistent input order

print(f"\nModel saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")
print(f"Feature names saved to: {feature_names_path}")

print("\nModel training script finished.")

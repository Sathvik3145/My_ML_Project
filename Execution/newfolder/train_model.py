import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 1: Load or Create the Dataset
try:
    df = pd.read_csv("autism_dataset.csv")  # Load your dataset if available
except FileNotFoundError:
    print("Dataset not found! Creating a synthetic dataset.")
    data = {
        'A1_Score': np.random.choice([0, 1], size=100),
        'A2_Score': np.random.choice([0, 1], size=100),
        'A3_Score': np.random.choice([0, 1], size=100),
        'A4_Score': np.random.choice([0, 1], size=100),
        'A5_Score': np.random.choice([0, 1], size=100),
        'A6_Score': np.random.choice([0, 1], size=100),
        'A7_Score': np.random.choice([0, 1], size=100),
        'A8_Score': np.random.choice([0, 1], size=100),
        'A9_Score': np.random.choice([0, 1], size=100),
        'A10_Score': np.random.choice([0, 1], size=100),
        'age': np.random.randint(5, 60, size=100),
        'gender': np.random.choice(['Male', 'Female'], size=100),
        'ethnicity': np.random.choice(['White-European', 'Black', 'Asian'], size=100),
        'jaundice': np.random.choice(['Yes', 'No'], size=100),
        'austim': np.random.choice(['Yes', 'No'], size=100),
        'used_app_before': np.random.choice(['Yes', 'No'], size=100),
        'relation': np.random.choice(['Self', 'Parent', 'Relative'], size=100),
        'Class/ASD': np.random.choice([0, 1], size=100)  # Target variable
    }
    df = pd.DataFrame(data)
    df.to_csv("autism_dataset.csv", index=False)

# Step 2: Preprocessing
X = df.drop(columns=['Class/ASD'])  # Features
y = df['Class/ASD']  # Target variable

label_encoders = {}  # Store encoders for later use
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

scaler = StandardScaler()  # Standardize numerical values
X_scaled = scaler.fit_transform(X)

# Step 3: Train SVC Model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Step 4: Save Model & Preprocessors
with open("svc_model.pkl", "wb") as f:
    pickle.dump((model, scaler, label_encoders), f)

print("Model training complete! SVC model saved as 'svc_model.pkl'.")

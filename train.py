import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load your dataset
df = pd.read_csv("final.csv")

# Ensure consistent column names if required
df.columns = df.columns.str.strip()  # Remove any extra spaces in column names
print("Training Data Columns:", df.columns)

# Step 2: Encode the target variable: 1 for Healthy, 0 for Unhealthy
df["Status"] = df["Status"].apply(lambda x: 1 if x == "Healthy" else 0)

# Step 3: Calculate class balance
healthy_count = df["Status"].sum()
unhealthy_count = len(df) - healthy_count
print(f"Healthy samples: {healthy_count}")
print(f"Unhealthy samples: {unhealthy_count}")

# Step 4: Separate features and target variable
X = df.drop("Status", axis=1)  # Features
y = df["Status"]  # Target (1 for Healthy, 0 for Unhealthy)

# Step 6: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for future use
joblib.dump(scaler, "scaler.pkl")

# Step 7: Calculate the min and max ranges for each feature (excluding 'Status')
feature_columns = ['Heart Rate', 'Movement', 'Oxygen Level', 'Temperature', 'Blood Pressure', 'Respiration Rate']
feature_ranges = {}

for col in feature_columns:
    feature_ranges[col] = {
        "min": df[col].min(),
        "max": df[col].max()
    }

# Save the feature ranges to a pickle file
joblib.dump(feature_ranges, "feature_ranges.pkl")

# Step 8: Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 9: Initialize models
models = {
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "SVM": SVC(random_state=42)
}

# Step 10: Train each model and evaluate accuracy
best_model = None
best_accuracy = 0

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy*100:.2f}%")

    # Check if this is the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Step 11: Save the best model
if best_model:
    joblib.dump(best_model, "best_model.pkl")
    print(f"\nThe best model is: {best_model.__class__.__name__} with accuracy: {best_accuracy*100:.2f}%")
else:
    print("No model performed better than a default classifier.")

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# Step 12: Store model accuracies for visualization
model_accuracies = {}

for model_name, model in models.items():
    # Predict and calculate accuracy for each model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[model_name] = accuracy

# Step 13: Plot the accuracies
plt.figure(figsize=(10, 6))
plt.bar(model_accuracies.keys(), [accuracy * 100 for accuracy in model_accuracies.values()], 
        color=['blue', 'green', 'orange', 'red'])
plt.title("Model Accuracies")
plt.ylabel("Accuracy (%)")
plt.xlabel("Models")
plt.ylim(80, 100)  # Set y-axis scale to start at 20%
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

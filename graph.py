import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
import seaborn as sns

# Load the model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load the dataset
df = pd.read_csv("final.csv")
df.columns = df.columns.str.strip()  # Remove extra spaces in column names
df["Status"] = df["Status"].apply(lambda x: 1 if x == "Healthy" else 0)

# Separate features and target
X = df.drop("Status", axis=1)
y = df["Status"]

# Scale the features
X_scaled = scaler.transform(X)

# Generate predictions
y_pred = model.predict(X_scaled)

# Accuracy Score
accuracy = accuracy_score(y, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}")

# 1. **Feature Importance**
if hasattr(model, "feature_importances_"):
    feature_importances = model.feature_importances_
    plt.barh(X.columns, feature_importances)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.show()
else:
    print("Model does not support feature importance.")

# 2. **Confusion Matrix**
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Unhealthy", "Healthy"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# 3. **Feature Distribution (Before Scaling)**
X.hist(bins=20, figsize=(12, 10))
plt.suptitle("Feature Distribution Before Scaling")
plt.show()

# 4. **Feature Distribution (After Scaling)**
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled_df.hist(bins=20, figsize=(12, 10))
plt.suptitle("Feature Distribution After Scaling")
plt.show()

# 5. **Actual vs Predicted Status**
plt.scatter(range(len(y)), y, label='Actual', color='green', marker='o')
plt.scatter(range(len(y_pred)), y_pred, label='Predicted', color='red', alpha=0.7)
plt.legend()
plt.xlabel("Samples")
plt.ylabel("Status (1=Healthy, 0=Unhealthy)")
plt.title("Actual vs Predicted Status")
plt.show()

# 6. **Sample Predictions Visualization**
sample_data = pd.DataFrame({
    'Heart Rate': [125, 80, 145, 115, 70],
    'Movement': [13, 4, 10, 9, 2],
    'Oxygen Level': [98, 85, 95, 97, 90],
    'Temperature': [36.6, 39.2, 36.7, 37.8, 39.1],
    'Blood Pressure': [120, 110, 130, 125, 115],
    'Respiration Rate': [18, 20, 16, 19, 22]
})

# Scale the sample data
sample_data_scaled = scaler.transform(sample_data)
sample_predictions = model.predict(sample_data_scaled)

sns.barplot(x=sample_data.index + 1, y=sample_predictions, palette="coolwarm")
plt.xticks(range(5), [f"Sample {i+1}" for i in range(5)])
plt.xlabel("Sample")
plt.ylabel("Prediction (1=Healthy, 0=Unhealthy)")
plt.title("Predictions for Manually Entered Samples")
plt.show()

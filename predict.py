import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd

# Load the trained model, scaler, and feature ranges
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_ranges = joblib.load("feature_ranges.pkl")

# Function to make predictions
def make_prediction():
    try:
        # Get values from the input fields
        heart_rate = float(entry_heart_rate.get())
        movement = float(entry_movement.get())
        oxygen_level = float(entry_oxygen_level.get())
        temperature = float(entry_temperature.get())
        blood_pressure = float(entry_blood_pressure.get())
        respiration_rate = float(entry_respiration_rate.get())

        # Check if values are within the calculated ranges
        out_of_range = False
        if not (feature_ranges['Heart Rate']['min'] <= heart_rate <= feature_ranges['Heart Rate']['max']):
            out_of_range = True
        if not (feature_ranges['Movement']['min'] <= movement <= feature_ranges['Movement']['max']):
            out_of_range = True
        if not (feature_ranges['Oxygen Level']['min'] <= oxygen_level <= feature_ranges['Oxygen Level']['max']):
            out_of_range = True
        if not (feature_ranges['Temperature']['min'] <= temperature <= feature_ranges['Temperature']['max']):
            out_of_range = True
        if not (feature_ranges['Blood Pressure']['min'] <= blood_pressure <= feature_ranges['Blood Pressure']['max']):
            out_of_range = True
        if not (feature_ranges['Respiration Rate']['min'] <= respiration_rate <= feature_ranges['Respiration Rate']['max']):
            out_of_range = True

        if out_of_range:
            # Show a generic message if any input is out of range
            messagebox.showinfo("Prediction Result", "The baby is Unhealthy. Please consult a doctor.")
            return

        # Create a DataFrame from the input values
        input_data = pd.DataFrame({
            'Heart Rate': [heart_rate],
            'Movement': [movement],
            'Oxygen Level': [oxygen_level],
            'Temperature': [temperature],
            'Blood Pressure': [blood_pressure],
            'Respiration Rate': [respiration_rate]
        })

        # Scale the input data
        input_scaled = scaler.transform(input_data)

        # Predict using the trained model
        prediction = model.predict(input_scaled)

        # Show the prediction result
        if prediction == 1:
            messagebox.showinfo("Prediction Result", "The baby is Healthy.")
        else:
            messagebox.showinfo("Prediction Result", "The baby is Unhealthy. Please consult a doctor.")

    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values for all fields.")

# Create the main window
root = tk.Tk()
root.title("Fetus Monitoring Prediction")

# Create labels and entry fields
label_heart_rate = tk.Label(root, text="Heart Rate:")
label_heart_rate.grid(row=0, column=0, padx=10, pady=5)
entry_heart_rate = tk.Entry(root)
entry_heart_rate.grid(row=0, column=1, padx=10, pady=5)

label_movement = tk.Label(root, text="Movement:")
label_movement.grid(row=1, column=0, padx=10, pady=5)
entry_movement = tk.Entry(root)
entry_movement.grid(row=1, column=1, padx=10, pady=5)

label_oxygen_level = tk.Label(root, text="Oxygen Level:")
label_oxygen_level.grid(row=2, column=0, padx=10, pady=5)
entry_oxygen_level = tk.Entry(root)
entry_oxygen_level.grid(row=2, column=1, padx=10, pady=5)

label_temperature = tk.Label(root, text="Temperature:")
label_temperature.grid(row=3, column=0, padx=10, pady=5)
entry_temperature = tk.Entry(root)
entry_temperature.grid(row=3, column=1, padx=10, pady=5)

label_blood_pressure = tk.Label(root, text="Blood Pressure:")
label_blood_pressure.grid(row=4, column=0, padx=10, pady=5)
entry_blood_pressure = tk.Entry(root)
entry_blood_pressure.grid(row=4, column=1, padx=10, pady=5)

label_respiration_rate = tk.Label(root, text="Respiration Rate:")
label_respiration_rate.grid(row=5, column=0, padx=10, pady=5)
entry_respiration_rate = tk.Entry(root)
entry_respiration_rate.grid(row=5, column=1, padx=10, pady=5)

# Create a button to make predictions
predict_button = tk.Button(root, text="Make Prediction", command=make_prediction)
predict_button.grid(row=6, column=0, columnspan=2, pady=10)

# Run the Tkinter event loop
root.mainloop()

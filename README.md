# Fetus Monitoring System

## Overview

The **Fetus Monitoring System (FMS)** is a machine learning-based tool developed to enhance prenatal care by classifying fetal health status as either healthy or unhealthy. This project leverages multiple machine learning algorithms and datasets containing fetal health indicators, ensuring reliable and accurate assessments for healthcare practitioners.

The system aims to reduce manual errors, improve diagnostic accuracy, and provide decision support to healthcare providers. The Gradient Boosting model was chosen for deployment due to its superior performance compared to other tested models.

---

## Project Structure

The repository contains the following files:

### 1. **Dataset and Preprocessing**
   - **`data/`**
     - Contains the dataset used for training and testing the model.
     - Includes fetal health indicators like FHR, accelerations, and decelerations.
   - **`preprocessing.py`**
     - Scripts for data cleaning, normalization, and feature engineering.

### 2. **Model Training**
   - **`models/`**
     - Contains various trained machine learning models (Logistic Regression, Random Forest, SVM, Gradient Boosting).
   - **`train_model.py`**
     - Script for training machine learning models and evaluating their performance using metrics such as accuracy, precision, recall, and F1-score.

### 3. **Evaluation**
   - **`evaluation.py`**
     - Includes evaluation metrics and tools for visualizing confusion matrices, ROC curves, and performance comparisons across models.

### 4. **Deployment**
   - **`app/`**
     - Files for deploying the Gradient Boosting model in a real-time environment.
     - Integration with clinical devices or applications for real-time predictions.
   - **`requirements.txt`**
     - Lists dependencies required to run the project and deploy the application.

### 5. **Documentation**
   - **`README.md`**
     - The current document explaining the project overview, structure, and functionality.
   - **`report/`**
     - A comprehensive project report with details on objectives, methodology, implementation, results, and conclusions.
   - **`diagrams/`**
     - Includes system architecture diagrams and workflow charts.

---

## Features

1. **Automated Fetal Health Classification**
   - Utilizes machine learning models to classify fetal health as "Healthy" or "Unhealthy."
2. **Data Preprocessing**
   - Ensures clean, optimized input data through feature selection and normalization.
3. **Multiple Model Evaluation**
   - Compares Logistic Regression, Random Forest, SVM, and Gradient Boosting to select the best-performing model.
4. **Explainability**
   - Incorporates SHAP values to provide insights into model predictions.
5. **Scalability and Usability**
   - Designed for diverse clinical environments with an intuitive interface.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fetus-monitoring-system.git
   ```
2. Navigate to the project directory:
   ```bash
   cd fetus-monitoring-system
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Data Preprocessing:**
   ```bash
   python preprocessing.py
   ```
2. **Train Models:**
   ```bash
   python train_model.py
   ```
3. **Evaluate Models:**
   ```bash
   python evaluation.py
   ```
4. **Run the Deployed Model:**
   ```bash
   python app/run.py
   ```

---

## Results

- The Gradient Boosting model achieved the highest accuracy, outperforming Logistic Regression and Random Forest models.
- Performance metrics:
  - **Accuracy:** 95%
  - **Precision:** 93%
  - **Recall:** 94%
  - **F1-Score:** 94%

---

## Future Enhancements

1. **Real-time Monitoring:**
   - Integrate with wearable devices for continuous tracking.
2. **Diverse Datasets:**
   - Expand to include datasets from various demographics to improve generalizability.
3. **Explainable AI:**
   - Enhance transparency using SHAP or LIME for detailed feature contributions.

---

## Contributors

- **Pasupuleti Lakshmi Sujith** (21BEC7429)
- **Anguluri Pavani Anvitha** (21BEC7498)
- **Pydisetty Sampath** (21BEC8466)
- **C Kishan Reddy** (21BEC7282)

**Guide:** Prof. M S Jagadeesh

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Let me know if you'd like further customization or any additional sections in the README!

# Fetus-Monitoring-System
Baby Monitoring During Pregnancy Time 
Fetus Monitoring System
Overview
The Fetus Monitoring System (FMS) is a machine learning-based tool developed to enhance prenatal care by classifying fetal health status as either healthy or unhealthy. This project leverages multiple machine learning algorithms and datasets containing fetal health indicators, ensuring reliable and accurate assessments for healthcare practitioners.

The system aims to reduce manual errors, improve diagnostic accuracy, and provide decision support to healthcare providers. The Gradient Boosting model was chosen for deployment due to its superior performance compared to other tested models.

Project Structure
The repository contains the following files:

1. Dataset and Preprocessing
data/
Contains the dataset used for training and testing the model.
Includes fetal health indicators like FHR, accelerations, and decelerations.
preprocessing.py
Scripts for data cleaning, normalization, and feature engineering.
2. Model Training
models/
Contains various trained machine learning models (Logistic Regression, Random Forest, SVM, Gradient Boosting).
train_model.py
Script for training machine learning models and evaluating their performance using metrics such as accuracy, precision, recall, and F1-score.
3. Evaluation
evaluation.py
Includes evaluation metrics and tools for visualizing confusion matrices, ROC curves, and performance comparisons across models.
4. Deployment
app/
Files for deploying the Gradient Boosting model in a real-time environment.
Integration with clinical devices or applications for real-time predictions.
requirements.txt
Lists dependencies required to run the project and deploy the application.
5. Documentation
README.md
The current document explaining the project overview, structure, and functionality.
report/
A comprehensive project report with details on objectives, methodology, implementation, results, and conclusions.
diagrams/
Includes system architecture diagrams and workflow charts.
Features
Automated Fetal Health Classification
Utilizes machine learning models to classify fetal health as "Healthy" or "Unhealthy."
Data Preprocessing
Ensures clean, optimized input data through feature selection and normalization.
Multiple Model Evaluation
Compares Logistic Regression, Random Forest, SVM, and Gradient Boosting to select the best-performing model.
Explainability
Incorporates SHAP values to provide insights into model predictions.
Scalability and Usability
Designed for diverse clinical environments with an intuitive interface.
Installation
Clone the repository:
bash
Copy
Edit
git clone https://github.com/your-username/fetus-monitoring-system.git
Navigate to the project directory:
bash
Copy
Edit
cd fetus-monitoring-system
Install the required dependencies:
bash
Copy
Edit
pip install -r requirements.txt
Usage
Data Preprocessing:
bash
Copy
Edit
python preprocessing.py
Train Models:
bash
Copy
Edit
python train_model.py
Evaluate Models:
bash
Copy
Edit
python evaluation.py
Run the Deployed Model:
bash
Copy
Edit
python app/run.py
Results
The Gradient Boosting model achieved the highest accuracy, outperforming Logistic Regression and Random Forest models.
Performance metrics:
Accuracy: 95%
Precision: 93%
Recall: 94%
F1-Score: 94%
Future Enhancements
Real-time Monitoring:
Integrate with wearable devices for continuous tracking.
Diverse Datasets:
Expand to include datasets from various demographics to improve generalizability.
Explainable AI:
Enhance transparency using SHAP or LIME for detailed feature contributions.
Contributors
Pasupuleti Lakshmi Sujith (21BEC7429)
Anguluri Pavani Anvitha (21BEC7498)
Pydisetty Sampath (21BEC8466)
C Kishan Reddy (21BEC7282)
Guide: Prof. M S Jagadeesh


# Credit Risk Prediction App
A machine learning web application that predicts whether a loan
applicant falls into the GOOD (lower risk) or BAD (higher risk)
category. The app uses a trained XGBoost classifier and a Streamlit
interface.
## Project Overview
Banks and lending organizations assess applicant details before making
lending decisions. This project demonstrates how machine learning helps
classify credit risk from applicant information.

The notebook is used for model development and evaluation. Decision
Tree, Random Forest, and XGBoost models are compared, and XGBoost is
selected as the final model for the application.

<p align="center">
  <img src="https://github.com/user-attachments/assets/fa4f15e6-dc9e-4949-ba69-12ed7938d18d"
       alt="Credit Risk Prediction Workflow Diagram"
       width="700">
</p>


## Features
- Predict credit risk for one applicant.
- Compare predictions for two applicants.
- Display the predicted class and model probability scores.
- Show the submitted applicant details.
- Download a single-applicant prediction report as CSV.
- Download a two-applicant comparison report as CSV.
- Load the trained model and categorical encoders from saved Joblib
  files.
  
## Input Features
The application uses these eight features:

  Feature            Description
  Age                Applicant's age
  Sex                Applicant's sex
  Job                Job level, from 0 to 3
  Housing            Housing status
  Saving accounts    Savings category
  Checking account   Checking account category
  Credit amount      Requested credit amount
  Duration           Credit duration in months

The app uses saved LabelEncoder objects for the categorical features
Sex, Housing, Saving accounts, and Checking account.

## Machine Learning Workflow
1. Load and inspect the credit risk dataset in the notebook.
2. Prepare the data and encode categorical features.
3. Split the data into training and testing sets.
4. Train and evaluate Decision Tree, Random Forest, and XGBoost models.
5. Select XGBoost as the final model.
6. Save the trained model and categorical encoders with Joblib.
7. Load the saved files in the Streamlit application.
8. Enter applicant details and generate a credit risk prediction.

## Technology Stack
- Python
- Pandas
- Scikit-learn
- XGBoost
- Joblib
- Streamlit
- VS Code

## Streamlit Application Output

The project includes an interactive web application built using Streamlit. Users enter applicant details, predict credit risk, and compare two applicants.

### 1. Single Applicant Prediction

Users enter the following details:
- Age
- Sex
- Job Level
- Housing
- Saving Accounts
- Checking Account
- Credit Amount
- Loan Duration

After clicking **Predict Credit Risk**, the application displays the predicted credit risk category.

- **GOOD (1):** The model classifies the applicant as lower credit risk.
- **BAD (0):** The model classifies the applicant as higher credit risk.

The application also displays the GOOD and BAD prediction scores, along with the applicant details. Users download the prediction report as a CSV file.

### 2. Compare Two Applicants

The application allows users to enter details for Applicant A and Applicant B.

It displays:
- Predicted credit risk for each applicant.
- Raw model prediction.
- GOOD prediction score for each applicant.
- A comparison table containing applicant details and prediction results.

Users download the comparison report as a CSV file.

### 3. Prediction Output

The trained XGBoost model processes the applicant's details and returns a predicted class. When supported by the model, the application also displays the predicted probabilities for both GOOD and BAD classes.

The results help users compare applicants based on the model's credit risk classification. These predictions do not guarantee loan repayment.

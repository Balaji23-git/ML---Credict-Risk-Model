# ML---Credict-Risk-Model

A **Machine Learning based Credit Risk Assessment System** that predicts the probability of loan default and generates a credit score for borrowers using financial and credit behavior features.

The application is deployed using **Streamlit**, allowing users to enter applicant details and instantly evaluate credit risk.

---

## Project Structure

- **frontend/**  
  Contains the **Streamlit web application** that allows users to input borrower and loan information and view prediction results.

- **backend/**  
  Contains the **prediction logic and trained machine learning model** used to estimate loan default probability and generate the credit score.

- **requirements.txt**  
  Lists all Python libraries required to run the application.

- **README.md**  
  Documentation explaining the project, setup instructions, and usage.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/loan-default-prediction.git
   cd loan-default-prediction
   ```
2. **Install dependencies:**
   ```commandline
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```commandline
   streamlit run app.py
   ```

# 📌 Project Overview

Financial institutions must assess borrower risk before approving loans.  
This project builds a **predictive model that estimates the probability of loan default** and converts it into a **credit score (300–900)** along with a **risk rating**.

The system analyzes borrower attributes such as:

- Income
- Loan amount
- Loan-to-income ratio
- Credit utilization
- Delinquency behavior
- Credit history

The result is presented in an **interactive Streamlit web application**.


# 🖥️ Application Interface

## Main Application Screen

![Application UI](./app_ui.PNG)

The application allows users to enter borrower and loan details and instantly view risk predictions.

---

## Prediction Results

![Prediction Results](./prediction_results.PNG)

The system returns:

- **Default Probability**
- **Credit Score**
- **Risk Rating**

---

# ⚙️ Features

✔ Predicts **loan default probability**  
✔ Generates **credit score (300–900 range)**  
✔ Assigns **risk rating categories**  
✔ Interactive **Streamlit UI**  
✔ Real-time **credit risk evaluation**  
✔ Clean dashboard-style interface

---

# 🧠 Machine Learning Pipeline

The project follows a typical **credit risk modeling pipeline**:

### 1️⃣ Data Preprocessing
- Handling missing values
- Feature engineering
- Creating financial ratios

### 2️⃣ Feature Engineering
Important engineered features include:

- **Loan-to-Income Ratio**
- **Average Days Past Due (DPD)**
- **Credit Utilization Ratio**
- **Delinquent Payment Ratio**

### 3️⃣ Handling Class Imbalance
Loan defaults are rare events.  
To improve model performance:

- **SMOTE-Tomek** was applied to balance classes.

### 4️⃣ Model Training
A **Logistic Regression model** was trained and optimized using:

- **Optuna Hyperparameter Optimization**

### 5️⃣ Model Evaluation

Metrics used:

- **AUC Score**
- **KS Statistic**
- **Recall for Default Class**

import streamlit as st
import joblib
import numpy as np
import pandas as pd

MODEL_PATH  = 'artifacts/model_data.joblib'

model_data = joblib.load(MODEL_PATH)
model = model_data['model']
features = model_data['features']
scaler = model_data['scaler']
cols_to_scale = model_data['cols_to_scale']


def prepare_df(age, income, loan_amount, loan_to_income_ratio, total_Loan_months, avg_Dpd,
            delinquent_ratio, credit_util_ratio, open_accounts, residence_type, loan_purpose, 
            loan_type):
    
    inputs = {
        'age': age,
        'loan_to_income': loan_to_income_ratio,
        'loan_tenure_months':total_Loan_months,
        'avg_dpd_per_deliquent':avg_Dpd,
        'delinquent_ratio': delinquent_ratio,
        'credit_utilization_ratio':credit_util_ratio,
        'number_of_open_accounts':open_accounts,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,
        
        ## Adding few more columns which were in our scaler
        'number_of_dependants': 1,
        'years_at_current_address': 1,
        'principal_outstanding': 1,
        'net_disbursement': 1,
        'processing_fee': 1,
        'gst':1,
        'sanction_amount': 1,
        'zipcode': 1,
        'bank_balance_at_application': 1,
        'number_of_closed_accounts': 1,
        'enquiry_count': 1

          
    }
    
    df = pd.DataFrame([inputs])
    
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    
    df = df[features]
    
    return df

def score_df(df):
    min_score = 300
    step_score = 600
    
    y= np.dot(df.values, model.coef_.T) + model.intercept_
    default_probability = 1 / (1 + np.exp(-y))
    non_default_probability = 1-default_probability
    
    credit_score = min_score + non_default_probability.flatten() *step_score
    
    def credit_rating(score):
        if 300<= score <500:
            return 'Poor'
        elif 500<= score <650:
            return 'Average'
        elif 650<= score <750:
            return 'Good'
        elif 750<= score <900:
            return 'Excellent'
        else:
            return 'Undefined'
        
    rating = credit_rating(credit_score[0])
    
    
    
    
    return default_probability.flatten()[0], int(credit_score[0]), rating


def predict(age, income, loan_amount, loan_to_income_ratio, total_Loan_months, avg_Dpd,
            delinquent_ratio, credit_util_ratio, open_accounts, residence_type, loan_purpose, 
            loan_type):
    
    df = prepare_df(age, income, loan_amount, loan_to_income_ratio, total_Loan_months, avg_Dpd,
            delinquent_ratio, credit_util_ratio, open_accounts, residence_type, loan_purpose, 
            loan_type)
    
    
    probability, credit_score, rating = score_df(df)
    
    
    return probability, credit_score, rating
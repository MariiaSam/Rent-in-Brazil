import streamlit as st
import joblib

import zipfile 
import os

import numpy as np
import pandas as pd


def load_models():
    linear_model = joblib.load('model/linear_regression_model.pkl')
    lasso_model = joblib.load('model/lasso_model.pkl')
    ridge_model = joblib.load('model/ridge_model.pkl')
    elastic_model = joblib.load('model/elastic_model.pkl')
    svr_model = joblib.load('model/svr_model.pkl')
    rf_model = joblib.load('model/random_forest_model.pkl')

    
    # with zipfile.ZipFile('model/random_forest_model.zip', 'r') as zip_ref: 
    #     zip_ref.extractall('model/random_forest_model')
    
    # rf_model =  joblib.load('model/random_forest_model/random_forest_model (2).pkl')
    
    
    return linear_model,  lasso_model,   ridge_model,  elastic_model, svr_model, rf_model


linear_model,  lasso_model, ridge_model,  elastic_model, svr_model, rf_model = load_models()


st.title("Models for predicting the value of rents in Brazil")

model_option = st.selectbox(
    "Select a model",
    ("LinearRegression", "Lasso", "Ridge", "ElasticNet", "SVR", "RandomForestRegressor" )
)
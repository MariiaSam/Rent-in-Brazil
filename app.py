import joblib
import numpy as np
import pandas as pd
import streamlit as st

def load_models():
    linear_model = joblib.load('model/linear_regression_model.pkl')
    lasso_model = joblib.load('model/lasso_model.pkl')
    ridge_model = joblib.load('model/ridge_model.pkl')
    elastic_model = joblib.load('model/elastic_model.pkl')
    svr_model = joblib.load('model/svr_model.pkl')
    rf_model = joblib.load('model/random_forest_model.pkl')
    return linear_model, lasso_model, ridge_model, elastic_model, svr_model, rf_model


def load_transformers():
    encoder = joblib.load('model/encoder.pkl')
    x_scaler = joblib.load('model/x_scaler.pkl')
    y_scaler = joblib.load('model/y_scaler.pkl')
    return encoder, x_scaler, y_scaler

def preprocess_data(df, encoder, x_scaler):
    columns_to_encode = ['animal', 'furniture', 'city']
    df[columns_to_encode] = df[columns_to_encode].astype(str)
    encoded_array = encoder.transform(df[columns_to_encode])
    encoded_columns = encoder.get_feature_names_out(columns_to_encode)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns, index=df.index)
    df = pd.concat([df.drop(columns=columns_to_encode), encoded_df], axis=1)
    df_scaled = pd.DataFrame(x_scaler.transform(df), columns=df.columns, index=df.index)
    return df_scaled

linear_model, lasso_model, ridge_model, elastic_model, svr_model, rf_model = load_models()
encoder, x_scaler, y_scaler = load_transformers()

st.title("Model for predicting the value of rents in Brazil")
st.write("Input data to get the Prediction Model")

area = st.number_input('Area (m²)', min_value=0, max_value=1000, value=50)
rooms = st.number_input('Rooms', min_value=0, max_value=10, value=3)
bathrooms = st.number_input('Bathrooms', min_value=0, max_value=10, value=1)
parking_spaces = st.number_input('Parking Spaces', min_value=0, max_value=10, value=1)
floor = st.number_input('Floor', min_value=0, max_value=100, value=3)
hoa = st.number_input('HOA (R$)', min_value=0, value=500)
property_tax = st.number_input('Property Tax (R$)', min_value=0, value=150)
fire_insurance = st.number_input('Fire Insurance (R$)', min_value=0, value=30)
animal = st.selectbox('Animal', ['acept', 'not acept'])
furniture = st.selectbox('Furniture', ['furnished', 'not furnished'])
city = st.selectbox('City', ['Belo Horizonte', 'Campinas', 'Porto Alegre', 'Rio de Janeiro', 'São Paulo'])

input_data = pd.DataFrame({
    'area': [area],
    'rooms': [rooms],
    'bathrooms': [bathrooms],
    'parking spaces': [parking_spaces], 
    'floor': [floor],
    'hoa': [hoa], 
    'property tax': [property_tax], 
    'fire insurance': [fire_insurance], 
    'animal': [animal],
    'furniture': [furniture],
    'city': [city]
})

preprocessed_input = preprocess_data(input_data, encoder, x_scaler)

model_option = st.selectbox("Select a model", ("LinearRegression", "Lasso", "Ridge", "ElasticNet", "SVR", "RandomForestRegressor"))

def predict_model(model, data):
    prediction = model.predict(data)
    return y_scaler.inverse_transform(prediction.reshape(-1, 1)).ravel()[0]

if model_option == "LinearRegression":
    prediction = predict_model(linear_model, preprocessed_input)
elif model_option == "Lasso":
    prediction = predict_model(lasso_model, preprocessed_input)
elif model_option == "Ridge":
    prediction = predict_model(ridge_model, preprocessed_input)
elif model_option == "ElasticNet":
    prediction = predict_model(elastic_model, preprocessed_input)
elif model_option == "SVR":
    prediction = predict_model(svr_model, preprocessed_input)
elif model_option == "RandomForestRegressor":
    prediction = predict_model(rf_model, preprocessed_input)

st.write(f"Predicted Rent Amount (R$): {prediction}")

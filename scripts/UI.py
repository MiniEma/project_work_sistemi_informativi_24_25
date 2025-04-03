import os       
import sys
sys.path.append(os.path.abspath('..'))
from src import config
import streamlit as st
import pandas as pd
import pickle

#loading the models
model_rf_hdn = pickle.load(open(os.path.join(config.MODEL_PATH, 'rf_model_hdn.pkl'), 'rb'))
model_lf_hdn = pickle.load(open(os.path.join(config.MODEL_PATH, 'lf_model_hdn.pkl'), 'rb'))
model_xb_hdn = pickle.load(open(os.path.join(config.MODEL_PATH, 'xb_model_hdn.pkl'), 'rb'))
model_rf_ll = pickle.load(open(os.path.join(config.MODEL_PATH, 'rf_model_ll.pkl'), 'rb'))
model_lf_ll = pickle.load(open(os.path.join(config.MODEL_PATH, 'lf_model_ll.pkl'), 'rb'))
model_xb_ll = pickle.load(open(os.path.join(config.MODEL_PATH, 'xb_model_ll.pkl'), 'rb'))

#creating user interface

st.title("House price Prediction")
st.write("This app predicts house prices based on different features")
st.write("NOTE: base input values are the mean values of the features")
#creating options on ll o hdn model
st.sidebar.title("Options")
st.sidebar.write("Select the model you want to use for prediction")
model_option = st.sidebar.selectbox("Select Model", ("Random Forest", "Linear Regression", "XGBoost"))
st.sidebar.write("Select the set of features you want to use for prediction")
features = st.sidebar.selectbox("Select Features",("Latitude,Longitude","House Age, Distance to nearest MRT Station, number of convenience stores") )

#creating the input fields for the features
if features == "Latitude,Longitude":
    latitude = st.number_input("Enter Latitude", min_value=-90.0, max_value=90.0, value=24.96)
    longitude = st.number_input("Enter Longitude", min_value=-180.0, max_value=180.0, value=121.53)
    input_data = pd.DataFrame({"latitude": [latitude], "longitude": [longitude]})
elif features == "House Age, Distance to nearest MRT Station, number of convenience stores":
    house_age = st.number_input("Enter House Age", min_value=0, max_value=100, value=17)
    distance = st.number_input("Enter Distance from nearest Metro/SubWay Station", min_value=0, max_value=10000, value=1083)
    stores = st.number_input("Enter number of convenience stores nearby", min_value=0, max_value=100, value=4)
    input_data = pd.DataFrame({"house_age": [house_age], "distance_to_nearest_MRT_station": [distance], "number_of_convenience_stores": [stores]})
else:
    st.write("Please select a feature set to proceed.")

#creating the button to predict the price
col1, col2 = st.columns([3,1])
with col1:
    pred_butt= st.button("Predict Price")
with col2:
    reset_butt= st.button("Reset")
if pred_butt :
    if model_option == "Random Forest":
        if features == "Latitude,Longitude":
            prediction = model_rf_ll.predict(input_data)
        elif features == "House Age, Distance to nearest MRT Station, number of convenience stores":
            prediction = model_rf_hdn.predict(input_data)
    elif model_option == "Linear Regression":
        if features == "Latitude,Longitude":
            prediction = model_lf_ll.predict(input_data)
        elif features == "House Age, Distance to nearest MRT Station, number of convenience stores":
            prediction = model_lf_hdn.predict(input_data)
    else:
        if features == "Latitude,Longitude":
            prediction = model_xb_ll.predict(input_data)
        elif features == "House Age, Distance to nearest MRT Station, number of convenience stores":
            prediction = model_xb_hdn.predict(input_data)

    st.markdown(
        f"<h2 style='font-size:36px; color:green;'>Predicted Price: {round(prediction[0], 2)*10}k TWD/PING</h2>",
    unsafe_allow_html=True
)

if reset_butt:
    st.rerun()  # This will reset the app and clear all inputs
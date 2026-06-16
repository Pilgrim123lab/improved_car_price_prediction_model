import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


#PAGE SETUP
st.set_page_config(
    page_title = 'Car2',
    layout = 'centered'
)

st.title = 'Car2'
st.markdown('Enter the car details to get an estimated price')
st.divider()

#LOAD THE MODEL
@st.cache_resource
def load_models():
    model = joblib.load('light_best_estimator.joblib')
    encoder = joblib.load('cat_encoder.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, encoder, scaler
#TRY TO LOAD THE MODEL, SHOW ERROR IF THE FILE IS NOT FOUND
try:
    model, encoder, scaler = load_models()
    st.success('models are loaded successfully')
except FileNotFoundError:
    st.error(
        "The models are not found in this file, make sure that all the files are loaded successfully,and retry"
    )
#CREATE AN INPUT FORM
st.subheader('Car specifications')

#LET US MALE TWO COLUMNS FOR BETTE READABILITY
col1, col2 = st.columns(2)
with col1:
    make = st.selectbox("Car Make", [
        'Toyota', 'Honda', 'Lexus', 'Mercedes-Benz', 'Ford', 'Nissan', 'Hyundai', 'Kia', 'Mazda', 'Acura', 'Volkswagen', 'BMW', 'Land Rover', 'Other', 'Infiniti', 'Peugeot', 'Chevrolet', 'Ponitac', 'Mitsubishi', 'Volvo', 'Audi', 'Dodge', 'Opel', 'Jaguar', 'Chrysler'
    ])
    condition = st.selectbox("Condition", [
        'Nigerian Used', 'Foreign Used', 'Brand New'
    ])
    transmission = st.selectbox("Transmission", [
        'Automatic', 'Manual', 'CVT'
    ])
    fuel = st.selectbox("Fuel", ['Petrol', 'Diesel', 'Electric', 'Hybrid'])
with col2:
    engine_size = st.number_input("Engine Size (cc)", min_value=800, max_value=6000, step=100)
    mileage = st.number_input("Mileage(Km)", min_value = 0,
                              max_value = 500000,
                              value = 25000,
                              step = 5000
                              )
    Car_Age = st.number_input("Car_Age(years)",
                              min_value=0,
                              max_value= 30,
                              value = 0,
                              step = 1
                              )
st.divider()

#PREDICTION BUTTON
predict_button = st.button("Predict Price", type = "primary",
                           use_container_width = True)
if predict_button:
    #we need to create  dataframe woth the user input in order to match it with our  model  parameter
    input_data = pd.DataFrame({
        'Make': [make],
        'Condition': [condition],
        'Fuel': [fuel],
        'Transmission': [transmission],
        'Mileage': [mileage],
        'Engine Size': [engine_size],
        'Car_age': [Car_Age]
    })
    #we need to encode the categorical variables and scale the numerical variables
    cat_columns = ['Make', 'Condition', 'Fuel', 'Transmission']
    num_columns = ['Mileage','Engine Size','Car_age']
    #apply one hot encoding to the categprical columns
    input_cat = encoder.transform(input_data[cat_columns])
    #apply scaling to the numerical columns
    input_num = scaler.transform(input_data[num_columns])
    #combine the tranformed features
    input_processed = np.hstack((input_num, input_cat))
    #make prediction on the log scaled of the target variable
    pred_log = model.predict(input_processed)
    #convert back to the normal price
    normal_price = np.exp(pred_log)[0]
    #display the result
    st.balloons()
    #create a nice card for the result
    st.subheader("Prediction Result")
    st.metric(label = "Estimated Price", value = f"₦{normal_price:,.2f}")
    #i want to put a footer that will contain some basic information about the project and also some informations about me like the 
    #link to the github repo where i am going to save the project files and also mayhe the link to my linkedin profile
    st.divider()
    linkedin = "https://www.linkedin.com/in/olanrewaju11"
    github =  "https://github.com/Pilgrim123lab/improved_car_price_prediction_model"
    st.caption(f"This is a car price prediction app built using Streamlit. The model is trained on a dataset of used cars and can predict the price of a car based on its specifications. For more information about the project, you can visit my GitHub repository: [{github}]({github}) or connect with me on LinkedIn: [{linkedin}]({linkedin})")
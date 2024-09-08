import streamlit as st
import joblib
import numpy as np

scaler = joblib.load('Scaler.pkl')
model = joblib.load('Model.pkl')

st.title('Real Estate Price Prediction App')

st.divider()

bed = st.number_input('Enter the number of bedrooms',value=2, step=1)
bath = st.number_input('Enter the number of bathrooms',value=1, step=1)
size = st.number_input('Enter the number of size',value=1000, step=50)

X = (bed,bath,size)

st.divider()

predictbutton= st.button('Predict')

st.divider()
if predictbutton:
    
    st.balloons()
    
    X1 = np.array(X)
    X_array = scaler.transform([X1])
    
    prediction = model.predict(X_array)[0]
    
    st.write(f'The Prediction is {prediction: .2f}')
    
    
else:
    'Please use the button for prediciton'
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pickle

# App title and introduction
st.write("""
# HHV Prediction of MSW
This app predicts the **Higher Heating Value (HHV) of Municipal Solid Waste (MSW)**!
""")
st.write('---')

# Display an image
image = Image.open('soil1.jpg')
st.image(image, use_column_width=True)

# Load dataset and display information
data = pd.read_csv('msw_all.csv')
req_col_names = ["C_Percentage", "H_Percentage", "N_Percentage", "S_Percentage", "O_Percentage", "HHV"]
curr_col_names = list(data.columns)

mapper = {curr_col_names[i]: req_col_names[i] for i in range(len(curr_col_names))}
data = data.rename(columns=mapper)

st.subheader('Data Information')
st.write(data.head())
st.write(data.isna().sum())

# Display correlation matrix
corr = data.corr()
st.write(corr)

# Sidebar for user input parameters
st.sidebar.header('Specify Input Parameters')
def get_input_features():
    C_Percentage = st.sidebar.slider('C_Percentage', 9.00, 92.00, 10.00)
    H_Percentage = st.sidebar.slider('H_Percentage', 2.00, 14.50, 3.00)
    N_Percentage = st.sidebar.slider('N_Percentage', 0.00, 10.00, 5.00)
    S_Percentage = st.sidebar.slider('S_Percentage', 0.00, 2.64, 2.00)
    O_Percentage = st.sidebar.slider('O_Percentage', 0.00, 48.62, 44.72)
    
    data_user = {
        'C_Percentage': C_Percentage,
        'H_Percentage': H_Percentage,
        'N_Percentage': N_Percentage,
        'S_Percentage': S_Percentage,
        'O_Percentage': O_Percentage
    }
    
    features = pd.DataFrame(data_user, index=[0])
    return features

df = get_input_features()

# Main panel to display specified input parameters
st.header('Specified Input Parameters')
st.write(df)
st.write('---')

# Load the scaler and the trained model
model = pickle.load(open('new (2).pkl', 'rb'))

# Standardize the input data using the loaded scaler
df_std = scaler.transform(df)

# Prediction
st.header('Prediction of HHV (MJ/kg)')
prediction = model.predict(df_std)
st.write(prediction[0])
st.write('---')

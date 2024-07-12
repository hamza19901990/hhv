import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import streamlit as st
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pickle

st.write("""
# HHV Prediction of MSW
This app predicts the **Higher Heating Value (HHV) of Municipal Solid Waste (MSW)**!
""")
st.write('---')

image = Image.open(r'soil.jpg')
st.image(image, use_column_width=True)

data = pd.read_csv(r"finalequtionsmars.csv")
req_col_names = ["C_Percentage", "H_Percentage", "N_Percentage", "S_Percentage", "O_Percentage", "HHV"]
curr_col_names = list(data.columns)

mapper = {}
for i, name in enumerate(curr_col_names):
    mapper[name] = req_col_names[i]

data = data.rename(columns=mapper)
st.subheader('Data Information')
st.write(data.head())
st.write(data.isna().sum())
corr = data.corr()
st.write(corr)

X = data.iloc[:, :-1]  # Features - All columns but last
y = data.iloc[:, -1]  # Target - Last Column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the GradientBoostingRegressor
model = GradientBoostingRegressor(learning_rate=0.5, n_estimators=100)
model.fit(X_train, y_train)

st.sidebar.header('Specify Input Parameters')
def get_input_features():
    C_Percentage = st.sidebar.slider('C_Percentage', 6.25, 1587.50, 671.10)
    H_Percentage = st.sidebar.slider('H_Percentage', 280.00, 2470.00, 1496.90)
    N_Percentage = st.sidebar.slider('N_Percentage', 640.00, 4608.00, 2509.18)
    S_Percentage = st.sidebar.slider('S_Percentage', 36.00, 284.00, 127.89)
    O_Percentage = st.sidebar.slider('O_Percentage', 22.16, 179.00, 44.72)
   
    
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

# Main Panel
st.header('Specified Input Parameters')
st.write(df)
st.write('---')

# Reads in saved classification model
load_clf = pickle.load(open('new(2).pkl', 'rb'))

st.header('Prediction of HHV (MJ/kg)')
# Apply model to make predictions
prediction = load_clf.predict(df)
st.write(prediction)
st.write('---') 

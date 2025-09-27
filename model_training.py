import pickle 
import streamlit as st
import numpy as np
import pandas as pd

pipe=pickle.load(open("pipe.pkl","rb"))
data=pickle.load(open("data.pkl","rb"))

st.title("Laptop Price Predictor")

# Company
Company=st.selectbox("Brand",data["Company"].unique())

# Type of laptop
TypeName=st.selectbox("Type of Laptop",data["TypeName"].unique())

# Ram
Ram=st.selectbox("RAM(in GB)",data["Ram"].unique())

# Weight
Weight=st.number_input("Weight of the Laptop)in Kg)")

#TouchScreen
Touch_Screen_choice=st.selectbox("Touch Screen",["No","Yes"])

if Touch_Screen_choice == 'Yes':
    Touch_Screen = 1
else:
    Touch_Screen = 0

#IPS
IPS_choice=st.selectbox("IPS",["No","Yes"])

if IPS_choice == 'Yes':
    IPS = 1
else:
    IPS = 0

# ScreenSize
screen_size=st.number_input("ScreenSize(in Inches)")

# resolution
resolution=st.selectbox("Screen Resolution",['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#Cpu
cpu=st.selectbox("Cpu Brand",data["Cpu_brand"].unique())

hdd=st.selectbox("HDD(in GB)",[0,32,128,500,1000,2000])

ssd=st.selectbox("SSD(in GB)",[0,8,16,32,64,128,240,256,512,768,1000,1024])

gpu=st.selectbox("Gpu Brand",data["Gpu_brand"].unique())

os=st.selectbox("Operating System",data["Operating_Sys"].unique())

if st.button("Predict Price"):
    query=pd.DataFrame({
        "Company": [Company],
        'TypeName': [TypeName],
            'Ram': [Ram],
            'Weight': [Weight],
            'Touch_Screen': [Touch_Screen],
            'IPS_Panel': [IPS],
            'PPI': [np.sqrt(int(resolution.split("x")[0])**2 + int(resolution.split("x")[1])**2)/screen_size],
            'Cpu_brand': [cpu],
            'HDD': [hdd],
            'SSD': [ssd],
            'Gpu_brand': [gpu],
            'Operating_Sys': [os]
    })
    
    prediction=pipe.predict(query)
    st.title(f"The Price for this configuration is  {np.exp(prediction)}")



















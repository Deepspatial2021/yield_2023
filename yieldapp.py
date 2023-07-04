import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import PIL
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


st.set_page_config(page_title='Agriverse Platform',layout="wide")

model = pickle.load(open('rfr_model.pkl', 'rb'))

#env_df=pd.read_csv("yield_data.csv")
env_df=pd.read_excel("new data.xlsx")
env_df=env_df.dropna()
#model=joblib.load("Farm Yield v2.sav")
#env_df=pd.read_excel("new data.xlsx")
#env_df=env_df.dropna()

season_dict={'Mustard':"Rabi",'Paddy':"Kharif",'Sugarcane':"Kharif", 'Wheat':"Rabi" }

#%%
html_temp = """
    <div style="background-color:#032863;padding:10px">
    <h2 style="color:white;text-align:center;">Agriverse Platform</h2>
    </div>
    <div style="background-color:white;padding:7px">
    <h2 style="color:black;text-align:center;font-size:30px; font-weight:bold">Farm Yield Prediction Model</h2>
    </div>
    <style>
    [data-testid="stAppViewContainer"]{
        background-image: url("https://images.unsplash.com/photo-1495107334309-fcf20504a5ab?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=870&q=80");
        background-size: cover;
        opacity: 0.8;
        }
    [data-testid="stHeader"]{
        background-color:rgba(255,255,255);
        }
    [data-testid="stToolbar"]{
        right=2rem;
        }
    </style>
    """
    
st.markdown(html_temp,unsafe_allow_html=True)
#st.title("Digital Agriculture Platform")
# Header Display
#st.markdown('<div style="background-color:#00609C;padding:7px"> <h2 style="color:white;text-align:center;">Farm Yield Predictor</h2> </div>',unsafe_allow_html=True)
#st.header("")

left,right=st.columns(2)
# Take Inputs
with right:
    crop=st.selectbox("Crop",['Mustard','Paddy','Sugarcane', 'Wheat' ])
    district=st.selectbox("District",("Dehradun", "Champawat"),disabled=True)
    block=st.selectbox("Block",("Vikasnagar", "Dalu"),disabled=True)
    village=st.selectbox("Village",env_df['Village Name'].unique())
    farm_num_df=env_df.groupby(['Village Name','Crop_Name'])['Farm_ID'].unique().reset_index()
    pivot_df=pd.pivot_table(farm_num_df,index="Village Name",columns='Crop_Name',values="Farm_ID")
    pivot_df_2=pivot_df.applymap(lambda z:z[:10])
    vlg_df=pivot_df_2.loc[village,:]
    farm_opts=np.append(np.append(vlg_df[0],vlg_df[1]),vlg_df[2])
    farm_opts=np.array(farm_opts,dtype=int)
    farm=st.selectbox("Farm Number",farm_opts)
    season=season_dict[crop]

# ['Area', 'Humidity', 'K', 'N', 'P', 'Precipitation', 'Temperature',
#        'Crop Name_Mustard', 'Crop Name_Paddy', 'Crop Name_Sugarcane',
#        'Crop Name_Wheat']


inp_df=env_df.set_index(['Village Name','Farm_ID'])
n=inp_df.loc[(village,farm),'N']
p=inp_df.loc[(village,farm),'P']
k=inp_df.loc[(village,farm),'K']
ph=inp_df.loc[(village,farm),'pH']
area=inp_df.loc[(village,farm),'Area (Hectares)']
rain=inp_df.loc[(village,farm),'Rainfall']    
temp=inp_df.loc[(village,farm),'Temperature']
humid=inp_df.loc[(village,farm),'Humidity']
if crop=="Mustard":
    crop_enc=[1,0,0,0]
elif crop=="Paddy":
    crop_enc=[0,1,0,0]
elif crop=="Sugarcane":
    crop_enc=[0,0,1,0]    
elif crop=="Wheat":
    crop_enc=[0,0,0,1]    


with left:
    col1,col2=st.columns(2)
    with col1:
        fig_temp = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = temp,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Temperature","font":{"size":26,"color":"white"}},  
        gauge = {
                'axis': {'range': [None, 40], 'tickwidth': 1, 'tickcolor': "darkred"},
                'bar': {'color': "#ccca49"},
                'bgcolor': "white",
                'borderwidth': 1,
                'bordercolor': "black",
                'steps': [
                    {'range': [0, 25], 'color': '#07448f'},
                    {'range': [25, 40], 'color': '#f01707'}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 38}}))
        fig_temp.update_layout(height=250, paper_bgcolor = "#306bb3", font = {'color': "white", 'family': "calibri"})
        st.plotly_chart(fig_temp,use_container_width=True)

    with col2:
        fig_humid = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = humid,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Humidity","font":{"size":26,"color":"white"}},
            gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#2fc2bf"},
                    'bgcolor': "white",
                    'borderwidth': 1,
                    'bordercolor': "black",
                    'steps': [
                        {'range': [0, 50], 'color': '#8f112c'},
                        {'range': [50, 100], 'color': '#191252'}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 98}}))
        fig_humid.update_layout(height=250, paper_bgcolor = "#306bb3", font = {'color': "white", 'family': "calibri"})
        st.plotly_chart(fig_humid,use_container_width=True)
       

    a1,a2,a3=st.columns(3)    
    with a2:
      crp = Image.open("imgs/"+str(crop)+".jpg")
      crp_1=crp.resize((200,150))
      st.image(crp_1)
    

if st.button("Calculate",use_container_width=True):
    inp_x=np.append([area,humid,k,n,p,rain,temp],crop_enc)
    farm_yield=np.round(model.predict([inp_x])[0],2)
    season_text="Season - "+season
    z1,z2=st.columns(2)
    with z1:
        st.subheader(season_text)
        string_2=("Crop Yield of "+str(crop) +" = "+str(farm_yield)+"(kg/ha)").title()
#        st.success('Crop Yield of  [ {} ] on your farm'.format(result))
#        st.subheader(":blue"[string_2])
        st.subheader(string_2)   

















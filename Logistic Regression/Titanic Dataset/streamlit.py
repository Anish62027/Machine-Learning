import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


st.header("Titanic Dataset Predictions")
st.sidebar.header("Chart Analysis")
titanic_model = 'C:\\Users\\Anish Avasthi\\Desktop\\Sunstone\\titanic.pkl'

loaded_model = joblib.load(titanic_model)






# Pclass	Sex	Age	SibSp	Parch	Fare	Embarked
Pclass = st.number_input("Pclass ",1,3)

Sex = st.selectbox("Sex",["Male","Female"])
Sex_dist = {"Male":0,"Female":1}
Sex = Sex_dist[Sex]

Age = st.number_input("Age",10,100)

SibSp = st.number_input("Sibling or Spouse",1,8)

Parch = st.number_input("Parent , Child",0,6)

Fare = st.number_input("Fare",30,600)

Embarked = st.selectbox("Embarked",['Southampton', 'Cherbourg', 'Queenstown'])
Embarked_dist = {'Southampton':0, 'Cherbourg':1, 'Queenstown':2}
Embarked = Embarked_dist[Embarked]

x = np.array([[Pclass,Sex,Age,SibSp,Parch,Fare,Embarked]])



predicted_value =  loaded_model.predict(x)

decode_dict = {1:"Yes", 0:"No"}

predicted_name = decode_dict[predicted_value[0]]



button = st.button("SUBMIT")

if button:
    st.text("Predictions")
    st.info(predicted_name)
    st.info(predicted_value)


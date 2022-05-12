from calendar import day_abbr
from curses import color_content
from matplotlib.pyplot import colorbar, colormaps, xlabel
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.express as px
st.set_page_config(page_title='Calory calculator ML model', layout='wide')

df = pd.read_csv('five_sec')
df0 = pd.read_csv('half_sec')
# Menu
with st.sidebar:
    menu = option_menu(menu_title='Contents', menu_icon = 'menu-up', options=['Home', 'EDA', 'Calory calculator'],
    icons = ['house', 'clipboard-data', 'activity'], orientation='vertical')

if menu == 'Home':    
    st.title("Creating a Machine learning model based calory calculator")

    st.markdown("Created by: _Abubakr Mamajonov, Sardorbek Zokirov_")
    image_1 = Image.open("artur-luczka-N1zRvlXf-IM-unsplash.jpg")
    st.image(image_1)
    st.markdown("As the second module of our Epicode is coming to an end, we have been tasked to build an application that can predict burnt calories based on the Machine Learning model predictions.")
    st.markdown("Our team, Abubakr and Sardorbek, has developed a ML model that predict the activity of the person from the input data that is collected through sensors of the device. We have tested the application on input data and will be presenting the calculator that is built based on the model's results.")
    st.markdown("The data we have used to train and test our model is collected by the University of Bologna: _Carpineti C., Lomonaco V., Bedogni L., Di Felice M., Bononi L., 'Custom Dual Transportation Mode Detection by Smartphone Devices Exploiting Sensor Diversity', in Proceedings of the 14th Workshop on Context and Activity Modeling and Recognition (IEEE COMOREA 2018), Athens, Greece, March 19-23, 2018 [Pre-print available]_")
    st.markdown("This application contains some Exploratory Data Analysis we have performed on the data we have and a calculator that can predict the amount of burnt calories in a given amount of time.")

elif menu == 'EDA':
    data_choice = st.sidebar.selectbox('Datasets', ['Five second balanced dataset', 'Half second balanced dataset'])
    if data_choice == 'Five second balanced dataset':
        graph_choice = st.sidebar.selectbox('Graphs', ['Balance of users', 'Balance of targets', 'Features'])
        if graph_choice == 'Balance of targets':
            df_u1 = df[df['user'] == 'U1']
            fig = px.bar(x=df['target'].value_counts().values, y=df['target'].value_counts().index, template='ggplot2',
                        labels = {"x":'Number of samples', 'y':'Mode of transportation'}, title='Balance of targets in 5-second balanced dataset', height=600, width=800)
            fig1 = px.bar(x=df_u1['target'].value_counts().values, y=df_u1['target'].value_counts().index, template='ggplot2', text_auto='.2s',
                        labels = {"x":'Number of samples', 'y':'Mode of transportation'}, title='Balance of targets from only U1 data in 5-second balanced dataset', height=600, width=800)
            st.write(fig, fig1)
        elif graph_choice == 'Balance of users':
            fig = px.bar(x=df['user'].value_counts(ascending=True).values, y=df['user'].value_counts(ascending=True).index, template='ggplot2',
                        labels = {"x":'Sample per user', 'y':'Users'}, title='Balance of samples from each user', height=600, width=800)
            st.write(fig)
        else:
            fig = px.imshow(df.isnull(), height=600, width=800, title='Missing (NaN) values in the dataset') # yticklabels=False, cbar=False,cmap='viridis'
            fig_1 = px.scatter(df, x = df['android.sensor.step_counter#mean'], y=df['android.sensor.accelerometer#mean'], color=df['target'])
            st.write(fig, fig_1)
    else:
        graph_choice = st.sidebar.selectbox('Graphs', ['Balance of users', 'Balance of targets', 'Features'])
        if graph_choice == 'Balance of targets':
            df_u1 = df0[df0['user'] == 'U1']
            fig = px.bar(x=df0['target'].value_counts().values, y=df0['target'].value_counts().index, template='ggplot2',
                        labels = {"x":'Number of samples', 'y':'Mode of transportation'}, title='Balance of targets in 0.5-second balanced dataset', height=600, width=800)
            fig1 = px.bar(x=df_u1['target'].value_counts().values, y=df_u1['target'].value_counts().index, template='ggplot2', text_auto='.2s',
                        labels = {"x":'Number of samples', 'y':'Mode of transportation'}, title='Balance of targets from only U1 data in 0.5-second balanced dataset', height=600, width=800)
            st.write(fig, fig1)
        elif graph_choice == 'Balance of users':
            fig = px.bar(x=df0['user'].value_counts(ascending=True).values, y=df0['user'].value_counts(ascending=True).index, template='ggplot2',
                        labels = {"x":'Sample per user', 'y':'Users'}, title='Balance of samples from each user', height=600, width=800)
            st.write(fig)
        else:
            fig = px.imshow(df0.isnull(), height=600, width=800, title='Missing (NaN) values in the dataset') # yticklabels=False, cbar=False,cmap='viridis'
            fig_1 = px.scatter(df0, x = df0['android.sensor.step_counter#mean'], y=df0['android.sensor.accelerometer#mean'], color=df0['target'])
            st.write(fig, fig_1)
else:
    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files = True)
    for uploaded_file in uploaded_files:
        data = pd.read_csv(uploaded_file)
        st.write("filename:", uploaded_file.name)
        st.write(data)
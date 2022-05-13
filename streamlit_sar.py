from calendar import day_abbr
#from curses import color_content
from matplotlib.pyplot import colorbar, colormaps, xlabel
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
import joblib
st.set_page_config(page_title='Calorie Calculator ML model', layout='wide')

ndf = pd.read_csv('new_data 2.csv')
ndf[['time', 'activityrecognition#1']] = ndf[['time', 'activityrecognition#1']].astype('int')
ndf['activityrecognition#1'] = ndf['activityrecognition#1'].astype('int')
df = pd.read_csv('five_sec')
df0 = pd.read_csv('half_sec')
# Menu
with st.sidebar:
    menu = option_menu(menu_title='Contents', menu_icon = 'menu-up', options=['Home', 'Data Analysis', 'Calorie Calculator'],
    icons = ['house', 'clipboard-data', 'activity'], orientation='vertical')

if menu == 'Home':    
    st.title("Creating a Machine learning model based Calorie Calculator")

    st.markdown("Created by: _Abubakr Mamajonov, Sardorbek Zokirov_")
    image_1 = Image.open("artur-luczka-N1zRvlXf-IM-unsplash.jpg")
    st.image(image_1)
    st.markdown("As the second module of our Epicode is coming to an end, we have been tasked to build an application that can predict burnt calories based on the Machine Learning model predictions.")
    st.markdown("Our team, Abubakr and Sardorbek, has developed a ML model that predict the activity of the person from the input data that is collected through sensors of the device. We have tested the application on input data and will be presenting the calculator that is built based on the model's results.")
    st.markdown("The data we have used to train and test our model is collected by the University of Bologna: _Carpineti C., Lomonaco V., Bedogni L., Di Felice M., Bononi L., 'Custom Dual Transportation Mode Detection by Smartphone Devices Exploiting Sensor Diversity', in Proceedings of the 14th Workshop on Context and Activity Modeling and Recognition (IEEE COMOREA 2018), Athens, Greece, March 19-23, 2018 [Pre-print available]_")
    st.markdown("This application contains some Exploratory Data Analysis we have performed on the data we have and a calculator that can predict the amount of burnt calories in a given amount of time.")

elif menu == 'Data Analysis':
    data_choice = st.sidebar.selectbox('Datasets', ['Five second balanced dataset', 'Half second balanced dataset'])
    if data_choice == 'Five second balanced dataset':
        graph_choice = st.sidebar.selectbox('Graphs', ['Balance of users', 'Balance of targets', 'Features'])
        if graph_choice == 'Balance of targets':
            df_u1 = df[df['user'] == 'U1']
            df2 = df[df.user != 'U1']
            # df2 = df.dropna(how='all')
            fig = px.bar(x=df['target'].value_counts().values, y=df['target'].value_counts().index, template='ggplot2',
                        labels = {"x":'Number of samples', 'y':'Mode of transportation'}, title='Balance of targets in 5-second balanced dataset', height=600, width=800)
            fig1 = px.bar(x=df_u1['target'].value_counts().values, y=df_u1['target'].value_counts().index, template='ggplot2', text_auto='.2s',
                        labels = {"x":'Number of samples', 'y':'Mode of transportation'}, title='Balance of targets from only U1 data in 5-second balanced dataset', height=600, width=800)
            fig2 = px.bar(x=df2['target'].value_counts().values, y=df2['target'].value_counts().index, template='ggplot2',
                        labels = {"x":'Number of samples', 'y':'Mode of transportation'}, title='Balance of targets for all other users in 5-second balanced dataset', height=600, width=800)
            st.write(fig, fig1, fig2)

        # elif graph_choice == 'Data Info':
        #     st.header('Data Information')
        #     st.text('Before feature selection and data cleaning data has n rows and m columns.\nAfter Feature selection and data cleaning data Thre are n rows and m columns in the data')
        #     st.write(ndf)
        #     st.text('Number of rows: ')
        #     st.text('Number of columns: ')
        #     #st.text('Target Information: 0 for doing activity and 1 for not doing activity')

        elif graph_choice == 'Balance of users':
            fig = px.bar(x=df['user'].value_counts(ascending=True).values, y=df['user'].value_counts(ascending=True).index, template='ggplot2',
                        labels = {"x":'Sample per user', 'y':'Users'}, title='Balance of samples from each user', height=600, width=800)
            st.write(fig)


            st.markdown('Users by target in 5 second dataset')
            st.write(plt.figure(figsize=(25, 12)),sb.countplot(x='user', hue='target',data=df.sort_values(by=['user'])),plt.legend(loc='upper right'))
        else:
            fig = px.imshow(df.isnull(), height=600, width=800, title='Missing (NaN) values in the dataset') # yticklabels=False, cbar=False,cmap='viridis'
            # fig_1 = px.scatter(df, x = df['android.sensor.step_counter#mean'], y=df['android.sensor.accelerometer#mean'], color=df['target'])
            st.write(fig)
    else:
        graph_choice = st.sidebar.selectbox('Data Info and Graphs', ['Data Info','Balance of users', 'Balance of targets', 'Features'])
        if graph_choice == 'Balance of targets':
            df_u1 = df0[df0['user'] == 'U1']
            df3 = df0[df0.user != 'U1']
            fig = px.bar(x=df0['target'].value_counts().values, y=df0['target'].value_counts().index, template='ggplot2',
                        labels = {"x":'Number of samples', 'y':'Mode of transportation'}, title='Balance of targets in 0.5-second balanced dataset', height=600, width=800)
            fig1 = px.bar(x=df_u1['target'].value_counts().values, y=df_u1['target'].value_counts().index, template='ggplot2', text_auto='.2s',
                        labels = {"x":'Number of samples', 'y':'Mode of transportation'}, title='Balance of targets from only U1 data in 0.5-second balanced dataset', height=600, width=800)
            fig2 = px.bar(x=df3['target'].value_counts().values, y=df3['target'].value_counts().index, template='ggplot2',
                        labels = {"x":'Number of samples', 'y':'Mode of transportation'}, title='Balance of targets for all other users in 0.5-second balanced dataset', height=600, width=800)

            st.write(fig, fig1, fig2)

        elif graph_choice == 'Data Info':
            st.header('Data Information')
            st.text(f'Before Feature Selection and Data Cleaning, Data has 62586 rows and 71 columns.\nAfter Feature selection and data cleaning data Thre are {ndf.shape[0]} rows and {ndf.shape[1]} columns in the data')
            st.write(ndf)
            st.text(f'Number of rows: {ndf.shape[0]}')
            st.text(f'Number of columns: {ndf.shape[1]}')
            st.text('Target Information: 1 for doing activity and 0 for not doing activity')

        elif graph_choice == 'Balance of users':
            fig = px.bar(x=df0['user'].value_counts(ascending=True).values, y=df0['user'].value_counts(ascending=True).index, template='ggplot2',
                        labels = {"x":'Sample per user', 'y':'Users'}, title='Balance of samples from each user', height=600, width=800)
            st.write(fig)

            st.markdown('Users by target in 0.5 second dataset')
            st.write(plt.figure(figsize=(25, 12)),sb.countplot(x='user', hue='target',data=df0.sort_values(by=['user'])),plt.legend(loc='upper right'))
        else:
            fig = px.imshow(df0.isnull(), height=600, width=800, title='Missing (NaN) values in the dataset') # yticklabels=False, cbar=False,cmap='viridis'
            # fig_1 = px.scatter(df0, x = df0['android.sensor.step_counter#mean'], y=df0['android.sensor.accelerometer#mean'], color=df0['target'])
            st.write(fig)
else:
    df_train = pd.read_csv("new_data 2.csv")
    X = df_train.drop(['target'], axis=1)
    y = df_train["target"]

    clf = RandomForestClassifier()
    clf.fit(X, y)

    joblib.dump(clf, "clf.pkl")
    st.title("Calorie Calculator")

    st.markdown("Please insert the following information: ")
    with st.form(key='my_form_to_submit'):
    
    
    # Input bar 1
        gender = st.number_input("Enter Gender: (1 is for female and 0 is for male)",step=0)
        age = st.number_input("Enter age:",step=0)
        weight = st.number_input("Enter weight: (in kg)",step=0)
        height = st.number_input("Enter height: (in cm)",step=0)
        timer = st.number_input("Enter time: (in minutes)",step=0)
        st.markdown("Please upload a file with data")

        uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
        for uploaded_file in uploaded_files:
            bytes_data = pd.read_csv(uploaded_file)

        submit_button = st.form_submit_button(label='Calculate')





    if submit_button:
        # Unpickle classifier
        clf = joblib.load("clf.pkl")
        
        # Store inputs into dataframe
        X = pd.DataFrame(bytes_data).values 
                       
        
        # Get prediction
        prediction = clf.predict(X)[0]
        
        # Output prediction
        if prediction == 1:
            
            # st.markdown(f'prediction {prediction}')
            met = 3.0
            # Using Harris Benedict equation
            time = timer / 60 #minutes
            cal = met * 3.5 * weight/200
            cal_h = cal * timer
            if cal_h == 0 :
                st.markdown('Wrong information entered, please check it one more')
                st.markdown("MET is calculated from the following source: https://www.omicsonline.org/articles-images/2157-7595-6-220-t003.html")
                
            else:
                st.markdown('Moderate physical activity')
                st.markdown(f'Calories burned in {int(timer)} minutes is: {cal_h}')
                st.markdown("MET is calculated from the following source: https://www.omicsonline.org/articles-images/2157-7595-6-220-t003.html")

        
        else:
            
            # st.markdown(f'prediction {prediction}')
            met = 1.0
            # Using Harris-Benedict equation
            time = timer / 60
            cal = met * 3.5 * weight/200
            cal_h = cal*timer
            if cal_h == 0:
                st.markdown('Wrong information entered, please check it once more')
                st.markdown("MET is calculated from the following source: https://www.omicsonline.org/articles-images/2157-7595-6-220-t003.html")
            else:
                st.markdown('Light intensity activity, e.g. being in a car, standing still, etc.') 
                st.markdown(f'Calories burned in {int(timer)} minutes is: {cal_h}')
                st.markdown("MET is calculated from the following source: https://www.omicsonline.org/articles-images/2157-7595-6-220-t003.html")
    

        

    

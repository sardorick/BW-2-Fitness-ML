import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("new_data.csv")
X = df.drop(['target'], axis=1)
y = df["target"]

clf = RandomForestClassifier()
clf.fit(X, y)

joblib.dump(clf, "clf.pkl")

# Title
st.header("Streamlit Machine Learning App")



        
with st.form(key='my_form_to_submit'):
    
    
    # Input bar 1
    gender = st.number_input("Enter Gender",step=0)

    age = st.number_input("Enter age",step=0)

    weight = st.number_input("Enter weight",step=0.0)

    height = st.number_input("Enter height",step=0.0)

    timer = st.number_input("Enter time",step=0.0)

    act_rec = st.number_input('Enter Activity recognition',step=0.0)

    and_sen_acc  = st.number_input("Enter android.sensor.accelerometer",step=0.0)

    and_sen_game_vec = st.number_input("Enter android.sensor.game_rotation_vector",step=0.0)

    and_sen_grav = st.number_input("Enter android.sensor.gravity",step=0.0)

    and_sen_gyro = st.number_input("Enter android.sensor.gyroscope",step=0.0)

    and_sen_gyro_unc = st.number_input("Enter android.sensor.gyroscope_uncalibrated",step=0.0)

    and_sen_lin_acc = st.number_input("Enter android.sensor.linear_acceleration",step=0.0)

    and_sen_mag_f = st.number_input("Enter android.sensor.magnetic_field",step=0.0)

    and_sen_mag_f_unc = st.number_input("Enter android.sensor.magnetic_field_uncalibrated",step=0.0)

    and_sen_ori = st.number_input("Enter android.sensor.orientation",step=0.0)

    and_sen_press = st.number_input("Enter android.sensor.pressure",step=0.0)

    and_sen_vec = st.number_input("Enter android.sensor.rotation_vector",step=0.0)

    and_sen_step_count = st.number_input("Enter android.sensor.step_counter",step=0.0)

    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()


    submit_button = st.form_submit_button(label='Submit')



if submit_button:
    # Unpickle classifier
    clf = joblib.load("clf.pkl")
    
    # Store inputs into dataframe
    X = pd.DataFrame([[bytes_data]], 
                     columns = ['time', 'activityrecognition#1', 'android.sensor.accelerometer#mean',
                                'android.sensor.game_rotation_vector#mean',
                                'android.sensor.gravity#mean', 'android.sensor.gyroscope#mean',
                                'android.sensor.gyroscope_uncalibrated#mean',
                                'android.sensor.linear_acceleration#mean',
                                'android.sensor.magnetic_field#mean',
                                'android.sensor.magnetic_field_uncalibrated#mean',
                                'android.sensor.orientation#mean', 'android.sensor.pressure#mean',
                                'android.sensor.rotation_vector#mean',
                                'android.sensor.step_counter#mean'])
    
    # Get prediction
    prediction = clf.predict(X)[0]
    
    # Output prediction
    if prediction == 1:
        st.text('Dude, you are GYM guy not Coder')
        st.text(f'prediction {prediction}')
        met = 2.8
        cal = age*weight*height*met
        st.text(f'Calorie burned: {cal}')

    
    else:
        st.text('Dude,go and do some activities') 
        st.text(f'prediction {prediction}')
        met = 1.1
        cal = age*weight*height*met
        st.text(f'Calorie burned: {cal}')
        

    
    
#timer, act_rec,and_sen_acc,and_sen_game_vec,and_sen_grav,and_sen_gyro,and_sen_gyro_unc,and_sen_lin_acc,and_sen_mag_f,and_sen_mag_f_unc,and_sen_ori,and_sen_press,and_sen_vec,and_sen_step_count
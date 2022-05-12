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

# Input bar 1
timer = st.number_input("Enter time",0)

act_rec = st.number_input('Enter Activity recognition',0)

and_sen_acc  = st.number_input("Enter android.sensor.accelerometer",0)

and_sen_game_vec = st.number_input("Enter android.sensor.game_rotation_vector",0)

and_sen_grav = st.number_input("Enter android.sensor.gravity",0)

and_sen_gyro = st.number_input("Enter android.sensor.gyroscope",0)

and_sen_gyro_unc = st.number_input("Enter android.sensor.gyroscope_uncalibrated",0)

and_sen_lin_acc = st.number_input("Enter android.sensor.linear_acceleration",0)

and_sen_mag_f = st.number_input("Enter android.sensor.magnetic_field",0)

and_sen_mag_f_unc = st.number_input("Enter android.sensor.magnetic_field_uncalibrated",0)

and_sen_ori = st.number_input("Enter android.sensor.orientation",0)

and_sen_press = st.number_input("Enter android.sensor.pressure",0)

and_sen_vec = st.number_input("Enter android.sensor.rotation_vector",0)

and_sen_step_count = st.number_input("Enter android.sensor.step_counter",0)


if st.button("Submit"):
    
    # Unpickle classifier
    clf = joblib.load("clf.pkl")
    
    # Store inputs into dataframe
    X = pd.DataFrame([[timer, act_rec,and_sen_acc,and_sen_game_vec,and_sen_grav,and_sen_gyro,and_sen_gyro_unc,
                      and_sen_lin_acc,and_sen_mag_f,and_sen_mag_f_unc,and_sen_ori,and_sen_press,and_sen_vec,and_sen_step_count]], 
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
        st.text('You are doing activities')
    else:
        st.text('You are not doing activities')

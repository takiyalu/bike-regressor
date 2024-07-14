import streamlit as st
import pandas as pd
import numpy as np
import pickle
def main():
    st.title(":bike: Bike Rental Predictor")
    calendar = st.sidebar.date_input('Select one date')
    workingday = st.sidebar.toggle('Working Day')
    holiday = st.sidebar.toggle('Holiday')
    month = calendar.month
    month_cos = np.cos(2*np.pi * month/12)
    month_sin = np.sin(2*np.pi * month/12)
    week_day = calendar.weekday()
    weekday_cos = np.cos(2*np.pi * week_day/7)
    weekday_sin = np.sin(2*np.pi * week_day/7)
    day = calendar.day
    day_cos = np.cos(2*np.pi * day/19)
    day_sin = np.sin(2*np.pi * day/19)
    temp = st.sidebar.slider('Temperature', min_value=0, max_value=100, step=1)
    atemp = st.sidebar.slider('Feels Like', min_value=0, max_value=100, step=1)
    humidity = st.sidebar.slider('Humidity', min_value=0, max_value=100, step=1)
    windspeed = st.sidebar.slider('Windspeed', min_value=0, max_value=60, step=1)
    
    def season_convert(s):
        if s == 'Spring':
            return 1
        elif s == 'Summer':
            return 2
        elif s == 'Fall':
            return 3
        elif s == 'Winter':
            return 4
    season = st.sidebar.radio('Choose one season', ['Spring','Summer','Fall','Winter'], index=0)
    season = season_convert(season)
    season_cos = np.cos(2*np.pi * season/4)
    season_sin = np.sin(2*np.pi * season/4)
    def weather_convert(w):
        if w == 'Good':
            return 1
        elif w == 'Mid':
            return 2
        elif w == 'Bad':
            return 3
        elif w == 'Very Bad':
            return 4
    weather = st.sidebar.radio('Choose weather condition', 
                                  ['Good',
                                   'Mid',
                                   'Bad',
                                   'Very Bad'], index=0)
    weather = weather_convert(weather)
    # holiday, workingday, weather, temp, atemp, humidity, windspeed, season_sin, month_sin, weekday_sin, day_sin, hour_sin
    data = []
    labels = ['holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity',
       'windspeed', 'season_sin', 'season_cos', 'month_sin', 'month_cos',
       'weekday_sin', 'weekday_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos']
    for hour in range(23):
        data_temp = [holiday, workingday, weather, temp, atemp, humidity, windspeed, season_sin, season_cos,month_sin, month_cos,weekday_sin,
                weekday_cos, day_sin, day_cos, np.cos(2*np.pi * hour/24), np.sin(2*np.pi * hour/24)]
        data.append(data_temp)
    df = pd.DataFrame(data, columns=labels)
    scaler = pickle.load(open('bike_scaler.pkl','rb'))
    model = pickle.load(open('bike_model.pkl', 'rb'))
    features_to_scale = ["temp", "atemp", 'humidity', 'windspeed', 'weather']
    df[features_to_scale] = scaler.transform(df[features_to_scale])
    st.write(df)
    predictions = model.predict(df)
    predictions = np.expm1(predictions)
    st.line_chart(predictions)
if __name__ =="__main__":
    main()
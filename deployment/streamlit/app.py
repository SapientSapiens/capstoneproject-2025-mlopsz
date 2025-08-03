import streamlit as st
import requests

st.set_page_config(page_title="Bike Sharing Demand Input Form")
st.title("🚲 Washington DC Bike Sharing Demand Hourly Prediction Service")

st.markdown("### Enter the following inputs:")

# Categorical Inputs
# season = st.selectbox("Season", options=["Spring", "Summer", "Fall", "Winter"], key="season")

weather = st.selectbox(
    "Weather", 
    options=["Clear/Partly Cloudy", "Mist/Cloudy", "Light Snow/Rain", "Heavy Precipitation"], 
    key="weather"
)

# Derived bad_weather (not shown to user now)
bad_weather_flag = int(weather in ["Light Snow/Rain", "Heavy Precipitation"])

holiday_flag= st.radio("Holiday", options=["Yes", "No"], key="holiday_display")
workingday_flag = st.radio("Working Day", options=["Yes", "No"], key="workingday_display")


hour = st.selectbox("Hour of Day", options=[str(i) for i in range(24)], key="hour")

day_name = st.selectbox(
    "Day Name", 
    options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], 
    key="day_name"
)

month = st.selectbox(
    "Month", 
    options=[
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ], 
    key="month"
)

year = st.selectbox("Year", options=["2021", "2012"], key="year")

# Numerical Inputs
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.01,  format="%.2f", key="humidity")
temp = st.number_input("Temperature (°C)", min_value=1.0, step=0.01,  format="%.2f",  key="temp")
atemp = st.number_input("Feels-like Temperature (°C)", min_value=1.0, step=0.01,  format="%.2f", key="atemp")
windspeed = st.number_input("Windspeed", min_value=0.0, step=0.001,  format="%.3f" , key="windspeed")

# Submit Button
if st.button("Submit for Prediction"):

    # Convert string flags to int
    holiday_flag = int(holiday_flag == "Yes")
    workingday_flag = int(workingday_flag  == "Yes")

    hour_int = int(hour)

    # bad_weather logic
    bad_weather_flag = int(weather in ["Light Snow/Rain", "Heavy Precipitation"])

    # hour_workingday
    hour_workingday = hour_int * workingday_flag

    # rush_hour
    rush_hour_flag = int(hour_int in [8, 16, 17, 18, 19])

    # part_of_day logic
    if hour_int <= 6:
        part_of_day = 'Early Morning'
    elif hour_int <= 11:
        part_of_day = 'Morning'
    elif hour_int <= 15:
        part_of_day = 'Afternoon'
    elif hour_int <= 19:
        part_of_day = 'Evening'
    else:
        part_of_day = 'Night'

    # temp_tolerance_feel
    if temp < 10:
        temp_tolerance_feel = 'brr_cold'
    elif temp < 20:
        temp_tolerance_feel = 'cold'
    elif temp < 29:
        temp_tolerance_feel = 'cool'
    elif temp < 39:
        temp_tolerance_feel = 'warm'
    elif temp <= 49:
        temp_tolerance_feel = 'hot'
    else:
        temp_tolerance_feel = 'boiling_hot'

    # temp_hum_factor
    temp_hum_factor = (temp * (100 - humidity)) / atemp

    # Derive season from month
    if month in ['December', 'January', 'February']:
        season = 'Winter'
    elif month in ['March', 'April', 'May']:
        season = 'Spring'
    elif month in ['June', 'July', 'August']:
        season = 'Summer'
    else:
        season = 'Fall'

    # Final payload
    payload = {
        "features": {
            "season": season,
            "weather": weather,
            "bad_weather": str(bad_weather_flag),
            "holiday": str(holiday_flag),
            "workingday": str(workingday_flag),
            "hour": str(hour_int),
            "hour_workingday": str(hour_workingday),
            "day_name": day_name,
            "month": month,
            "year": year,
            "rush_hour": str(rush_hour_flag),
            "part_of_day": part_of_day,
            "temp_tolerance_feel": temp_tolerance_feel,
            "humidity": humidity,
            "temp": temp,
            "windspeed": windspeed,
            "temp_hum_factor": round(temp_hum_factor, 4)
        }
    }
    #st.json(payload)

    try:
        # response = requests.post("http://localhost:8010/predict", json=payload)
        # response = requests.post("http://ec2-16-16-201-186.eu-north-1.compute.amazonaws.com:8010/predict", json=payload)
        response = requests.post("http://fastapi_app:8010/predict", json=payload) # call with container name mentioned in the docker-compose.yml

        if response.status_code == 200:
            result = response.json()
            prediction = result.get("prediction", "No prediction returned.")
            st.success(f"🔮 Predicted Bike Demand For That Hour :  **{prediction}**")
        else:
            st.error(f"❌ Prediction API error: {response.status_code}")
            st.text(response.text)

    except Exception as e:
        st.error(f"⚠️ Failed to connect to prediction API:\n{str(e)}")

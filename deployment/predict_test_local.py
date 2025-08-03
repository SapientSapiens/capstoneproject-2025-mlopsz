import requests

payload = {
    "features": {
        "season": "Summer",
        "weather": "Clear/Partly Cloudy",
        "bad_weather": "0",
        "holiday": "0",
        "workingday" : "1",
        "hour" : "21",
        "hour_workingday" : "21",
        "day_name" :  "Friday",
        "month" : "August",
        "year" : "2012",
        "rush_hour" : "0",
        "part_of_day" : "Night",
        "temp_tolerance_feel" : "warm",
        "humidity" : 58,
        "temp" : 31.16,
        "windspeed" : 12.998,
        "temp_hum_factor": 36.756635
    }
}


response = requests.post("http://localhost:8010/predict", json=payload)
print(response.json())
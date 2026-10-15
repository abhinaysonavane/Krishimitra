from flask import Flask, render_template, jsonify, request
import serial, time, threading
import pandas as pd
import joblib
import requests

app = Flask(__name__)

# 🔑 WEATHER API KEY
API_KEY = "paste here weather api key"

# Load trained models
crop_model = joblib.load('models/saved/crop_model.pkl')
irrigation_model = joblib.load('models/saved/irrigation_model.pkl')
crop_encoder = joblib.load('models/saved/crop_encoder.pkl')
irrigation_encoder = joblib.load('models/saved/irrigation_encoder.pkl')

# Arduino connection
try:
    arduino = serial.Serial('COM5', 9600, timeout=1)
    time.sleep(2)
except:
    arduino = None

sensor_data = {
    'temp': 0,
    'hum': 0,
    'soil': 0,
    'crop': 'N/A',
    'advice': 'N/A'
}

# 🌦️ Fetch weather
def get_weather(city):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
    try:
        return requests.get(url).json()
    except:
        return None

# 🌦️ Generate weather advice
def generate_weather_advice(weather):
    if weather is None:
        return ["⚠️ Weather data not available"]

    c = weather['current']
    temp = c['temp_c']
    humidity = c['humidity']
    condition = c['condition']['text'].lower()
    rain = c['precip_mm']

    advice = []

    if "rain" in condition or rain > 0:
        advice.append("🌧️ Rain expected → Do NOT irrigate")

    if temp > 30:
        advice.append("☀️ High temperature → Increase irrigation")

    if humidity < 60 and rain == 0:
        advice.append("🌾 Good weather → Suitable for harvesting")

    if humidity > 85:
        advice.append("⚠️ High humidity → Risk of crop disease")

    if not advice:
        advice.append("✅ Weather conditions normal")

    return advice

# Arduino background thread
def read_arduino():
    while arduino:
        try:
            line = arduino.readline().decode().strip()
            if line:
                t, h, s = map(float, line.split(','))

                sensor_data['temp'] = t
                sensor_data['hum'] = h
                sensor_data['soil'] = s

                df = pd.DataFrame([[t, h, s]],
                                  columns=['Temperature','Humidity','SoilMoisture'])

                crop_pred = crop_model.predict(df)
                irrigation_pred = irrigation_model.predict(df)

                sensor_data['crop'] = crop_encoder.inverse_transform(crop_pred)[0]
                sensor_data['advice'] = irrigation_encoder.inverse_transform(irrigation_pred)[0]

        except:
            continue

if arduino:
    threading.Thread(target=read_arduino, daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    city = request.args.get('city', 'Pune')

    weather = get_weather(city)
    alerts = generate_weather_advice(weather)

    return jsonify({
        **sensor_data,
        "weather_alerts": alerts
    })

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
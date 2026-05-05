import numpy as np
import pandas as pd
import os
import pickle
import requests
import json
import time
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "YOUR_API_KEY_HERE")


from flask_cors import CORS

from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
def normalize_weather(w):
    w = str(w).lower()

    if "rain" in w:
        return "Rainy"
    elif "fog" in w or "mist" in w or "haze" in w:
        return "Fog"
    else:
        return "Clear"

# =========================
# STEP 1: DATA GENERATION
# =========================


n = 3000

# =========================
# ALWAYS CREATE BASE DATASET (USED AT RUNTIME)
# =========================

time_vals = np.random.randint(0, 24, n)
day_type = np.random.choice(["Weekday", "Weekend"], n)
weather = np.random.choice(["Clear", "Rainy", "Fog"], n)
event_day = np.random.choice([0, 1], n)

passengers = []
flight_count = []
avg_luggage = []

for t, d, w, e in zip(time_vals, day_type, weather, event_day):
    if 6 <= t <= 10 or 17 <= t <= 21:
        p = np.random.randint(1000, 2000)
    else:
        p = np.random.randint(200, 1000)

    if d == "Weekend":
        p += 300
    if e == 1:
        p += 400
    if w == "Rainy":
        p -= 200
    elif w == "Fog":
        p -= 300

    passengers.append(max(100, p))
    flight_count.append(int(p / 50))
    avg_luggage.append(np.random.randint(1, 5))

crowd = []
for p in passengers:
    if p > 1500:
        crowd.append("High")
    elif p > 700:
        crowd.append("Medium")
    else:
        crowd.append("Low")

df = pd.DataFrame({
    "time": time_vals,
    "passengers": passengers,
    "day_type": day_type,
    "weather": weather,
    "event_day": event_day,
    "flight_count": flight_count,
    "avg_luggage": avg_luggage,
    "crowd": crowd
})

if not os.path.exists("crowd_model.pkl") or not os.path.exists("delay_model.pkl"):
    print("Training model...")

    time_vals = np.random.randint(0, 24, n)
    day_type = np.random.choice(["Weekday", "Weekend"], n)
    weather = np.random.choice(["Clear", "Rainy", "Fog"], n)
    event_day = np.random.choice([0, 1], n)

    passengers = []
    flight_count = []
    avg_luggage = []

    for t, d, w, e in zip(time_vals, day_type, weather, event_day):

        if 6 <= t <= 10 or 17 <= t <= 21:
            p = np.random.randint(1000, 2000)
        else:
            p = np.random.randint(200, 1000)

        if d == "Weekend":
            p += 300

        if e == 1:
            p += 400

        if w == "Rainy":
            p -= 200
        elif w == "Fog":
            p -= 300

        passengers.append(max(100, p))
        flight_count.append(int(p / 50))
        avg_luggage.append(np.random.randint(1, 5))

    crowd = []
    for p in passengers:
        if p > 1500:
            crowd.append("High")
        elif p > 700:
            crowd.append("Medium")
        else:
            crowd.append("Low")

    df = pd.DataFrame({
        "time": time_vals,
        "passengers": passengers,
        "day_type": day_type,
        "weather": weather,
        "event_day": event_day,
        "flight_count": flight_count,
        "avg_luggage": avg_luggage,
        "crowd": crowd
    })

    le_weather = LabelEncoder()
    df["weather"] = le_weather.fit_transform(df["weather"])

    delay_label = []
    for p in passengers:
        if p > 1500:
            delay_label.append("High")
        elif p > 800:
            delay_label.append("Medium")
        else:
            delay_label.append("Low")

    df["delay_label"] = delay_label

    le_day = LabelEncoder()
    le_crowd = LabelEncoder()
    df["day_type"] = le_day.fit_transform(df["day_type"])
    df["crowd"] = le_crowd.fit_transform(df["crowd"])

    le_delay = LabelEncoder()
    df["delay_label"] = le_delay.fit_transform(df["delay_label"])

    X = df[["time", "passengers", "day_type", "weather", "event_day", "flight_count", "avg_luggage"]]
    y = df["crowd"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    X_delay = df[["time", "passengers", "day_type", "weather", "event_day", "flight_count", "avg_luggage"]]
    y_delay = df["delay_label"]

    delay_model = RandomForestClassifier()
    delay_model.fit(X_delay, y_delay)

    pickle.dump(model, open("crowd_model.pkl", "wb"))
    pickle.dump(le_day, open("le_day.pkl", "wb"))
    pickle.dump(le_crowd, open("le_crowd.pkl", "wb"))
    pickle.dump(le_weather, open("le_weather.pkl", "wb"))
    pickle.dump(delay_model, open("delay_model.pkl", "wb"))
    pickle.dump(le_delay, open("le_delay.pkl", "wb"))

    print("Model trained & saved successfully!")

# =========================
# STEP 5: FLASK API
# =========================

app = Flask(__name__)
CORS(app)

# Home route to check server status
@app.route("/")
def home():
    return "Airport AI Backend is Running 🚀"

# Load saved model
model = pickle.load(open("crowd_model.pkl", "rb"))
le_day = pickle.load(open("le_day.pkl", "rb"))
le_crowd = pickle.load(open("le_crowd.pkl", "rb"))
le_weather = pickle.load(open("le_weather.pkl", "rb"))
delay_model = pickle.load(open("delay_model.pkl", "rb"))
le_delay = pickle.load(open("le_delay.pkl", "rb"))

from flask import send_from_directory


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    time_input = data["time"]
    passengers_input = data["passengers"]
    day_input = data["day_type"]

    # Encode input
    day_encoded = le_day.transform([day_input])[0]

    # Get real weather for better prediction
    weather_str = data.get("weather", "Clear")
    try:
        weather_clean = normalize_weather(weather_str)
        weather_encoded = le_weather.transform([weather_clean])[0]
    except:
        weather_encoded = 0

    # Default engineered features
    flight_count = int(passengers_input / 50)
    avg_luggage = 2
    event_day = data.get("event_day", 0)

    prediction = model.predict([[
        time_input,
        passengers_input,
        day_encoded,
        weather_encoded,
        event_day,
        flight_count,
        avg_luggage
    ]])
    result = le_crowd.inverse_transform(prediction)[0]

    return jsonify({
        "predicted_crowd": result
    })

# =========================
# AVIATION QUERY FILTER
# =========================
def is_aviation_query(msg):
    keywords = [
        "airport","flight","delay","weather","runway",
        "aircraft","traffic","passenger","crowd",
        "visibility","schedule","radar","atc"
    ]
    return any(word in msg.lower() for word in keywords)

# =========================
# AIRPORT EXTRACTION ENGINE
# =========================
def extract_airport(msg):
    msg = msg.lower().strip()

    # =========================
    # 1) MATCH ICAO CODES
    # =========================
    for name, info in AIRPORTS.items():
        icao = info["icao"]
        if icao.lower() in msg:
            return name

    # =========================
    # 2) MATCH FULL AIRPORT NAMES
    # =========================
    for name in AIRPORTS.keys():
        if name.lower() in msg:
            return name

    # =========================
    # 3) MATCH CITY NAMES (SMART)
    # =========================
    # Direct city mapping (if present)
    for ap_name, info in AIRPORTS.items():
        city = info["city"]

        if city.lower() in msg:
            return ap_name

    # =========================
    # 3.5) SPECIAL AIRPORT ALIASES
    # =========================
    aliases = {
        "heathrow": "Heathrow",
        "new york": "JFK",
        "los angeles": "LAX"
    }

    for alias, airport_name in aliases.items():
        if alias in msg:
            return airport_name

    # =========================
    # 4) FALLBACK: PARTIAL MATCH (VERY IMPORTANT)
    # =========================
    for name in AIRPORTS.keys():
        tokens = name.lower().replace("airport", "").split()
        for token in tokens:
            if token and token in msg:
                return name

    return None

# =========================
# INTELLIGENT CHAT ENGINE
# =========================

conversation_memory = {}

# =========================
# ADVANCED ADAPTIVE LEARNING
# =========================

import pickle as pkl

ADAPTIVE_FILE = "adaptive_state.pkl"

# =========================
# FEEDBACK DATASET STORAGE (ML RETRAINING)
# =========================
FEEDBACK_DATA_FILE = "feedback_data.csv"

# Load or initialize feedback dataset
if os.path.exists(FEEDBACK_DATA_FILE):
    feedback_df = pd.read_csv(FEEDBACK_DATA_FILE)
else:
    feedback_df = pd.DataFrame(columns=[
        "time","passengers","day_type","weather",
        "event_day","flight_count","avg_luggage","crowd","delay_label"
    ])

# Load previous state if exists
if os.path.exists(ADAPTIVE_FILE):
    with open(ADAPTIVE_FILE, "rb") as f:
        adaptive_state = pkl.load(f)
else:
    adaptive_state = {
        "global": {"passenger_bias": 0, "delay_bias": 0},
        "airports": {},
        "history": []
    }

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        message = data.get("message", "").lower()

        # =========================
        # GREETING + CONVERSATIONAL HANDLING
        # =========================
        greetings = ["hi", "hello", "hey", "good morning", "good evening"]
        if any(f" {greet} " in f" {message} " for greet in greetings):
            return jsonify({
                "reply": "Hello! I'm your airport intelligence assistant ✈️. I can help you with crowd levels, flight delays, risks, and airport insights. How can I assist you today?"
            })

        if " how are you " in f" {message} ":
            return jsonify({
                "reply": "I'm operating optimally and ready to assist you with real-time airport intelligence 😊. What would you like to know?"
            })

        import re

        # =========================
        # CONTEXT MEMORY
        # =========================
        airport_key = extract_airport(message)
        if not airport_key:
            airport_key = conversation_memory.get("last_airport", "Delhi")
        conversation_memory["last_airport"] = airport_key

        # =========================
        # LOAD AIRPORT DATA
        # =========================
        filename = airport_key.replace(" ", "_") + ".csv"
        filepath = os.path.join("airport_data", filename)

        if os.path.exists(filepath):
            local_df = pd.read_csv(filepath)
        else:
            local_df = df.copy()

        # =========================
        # TIME EXTRACTION
        # =========================
        time_input = 12
        match = re.search(r"\d+", message)
        if match:
            time_input = int(match.group())

        if "pm" in message and time_input < 12:
            time_input += 12
        if "am" in message and time_input == 12:
            time_input = 0

        # =========================
        # DYNAMIC PASSENGER ESTIMATION (SMART)
        # =========================
        try:
            passengers_input = int(
                local_df[local_df["time"] == time_input]["passengers"].mean()
            ) + adaptive_state["global"]["passenger_bias"] + adaptive_state["airports"].get(airport_key, {}).get("passenger_bias", 0)
            if np.isnan(passengers_input):
                raise Exception()
        except:
            if 17 <= time_input <= 21:
                passengers_input = 1600
            elif 6 <= time_input <= 10:
                passengers_input = 1300
            else:
                passengers_input = 700

        # =========================
        # ENCODING
        # =========================
        day_encoded = le_day.transform(["Weekday"])[0]

        weather_val = local_df.sample(1)["weather"].values[0]
        if isinstance(weather_val, (int, float)):
            weather_encoded = int(weather_val)
        else:
            try:
                weather_clean = normalize_weather(weather_val)
                weather_encoded = le_weather.transform([weather_clean])[0]
            except:
                weather_encoded = 0

        event_day = 0
        flight_count = int(passengers_input / 50)
        avg_luggage = 2

        # =========================
        # ADVANCED INTENT KEYWORDS
        # =========================

        crowd_keywords = [
            "crowd","traffic","busy","rush","packed",
            "congestion","queue","long lines",
            "passenger load","footfall","movement",
            "airport load","terminal load",
            "overcrowded","heavy traffic",
            "how busy","peak traffic",
            "passenger density"
        ]

        delay_keywords = [
            "delay","late","hold","waiting",
            "departure issue","arrival issue",
            "boarding issue","takeoff delay",
            "landing delay","flight disruption",
            "stuck","cancel","postponed",
            "slow operations","air traffic delay"
        ]

        risk_keywords = [
            "risk","danger","unsafe",
            "critical","emergency",
            "operational risk",
            "landing risk",
            "weather risk",
            "safety issue",
            "atc concern",
            "hazard","threat"
        ]

        peak_keywords = [
            "peak","rush hour","busiest time",
            "highest traffic","when busiest",
            "busy hours","traffic peak",
            "maximum crowd","heavy period"
        ]

        explain_keywords = [
            "why","explain","reason",
            "how","cause","what caused",
            "why is","why are",
            "analysis","justify"
        ]

        info_keywords = [
            "about","information","details",
            "tell me about","airport info",
            "runways","terminals",
            "iata","icao","famous",
            "where is","located",
            "airport overview",
            "airport statistics"
        ]

        # =========================
        # NEW INTENT DETECTION (MULTI-INTENT)
        # =========================
        def detect_intents(message):
            intents = []

            if any(k in message for k in crowd_keywords):
                intents.append("crowd")

            if any(k in message for k in delay_keywords):
                intents.append("delay")

            if any(k in message for k in risk_keywords):
                intents.append("risk")

            if any(k in message for k in peak_keywords):
                intents.append("peak")

            if any(k in message for k in explain_keywords):
                intents.append("explain")

            if any(k in message for k in info_keywords):
                intents.append("info")

            return intents

        intents = detect_intents(message)
        response = []

        # =========================
        # AIRPORT INFORMATION ENGINE
        # =========================
        if "info" in intents:
            info = AIRPORT_INFO.get(airport_key)

            if info:
                if "icao" in message:
                    reply = f"🛰 ICAO code for {airport_key}: {info['icao']}"
                elif "iata" in message:
                    reply = f"🛫 IATA code for {airport_key}: {info['iata']}"
                elif "runway" in message:
                    reply = f"🛬 {airport_key} has {info['runways']} runways"
                elif "terminal" in message:
                    reply = f"🏢 {airport_key} has {info['terminals']} terminals"
                elif "where" in message or "located" in message:
                    reply = f"📍 {airport_key} is located in {info['city']}, {info['country']}"
                elif "famous" in message:
                    reply = f"⭐ {airport_key}: {info['famous_for']}"
                else:
                    reply = f"""
✈️ {info['full_name']}

📍 City: {info['city']}, {info['country']}
🛫 IATA: {info['iata']}
🛰 ICAO: {info['icao']}
🏢 Terminals: {info['terminals']}
🛬 Runways: {info['runways']}
🌍 Type: {info['type']}
👥 Annual Traffic: {info['annual_passengers']}
⭐ {info['famous_for']}
"""
                return jsonify({"reply": reply})

        # =========================
        # CROWD PREDICTION
        # =========================
        if "crowd" in intents:
            input_df = pd.DataFrame([{
                "time": time_input,
                "passengers": passengers_input,
                "day_type": day_encoded,
                "weather": weather_encoded,
                "event_day": event_day,
                "flight_count": flight_count,
                "avg_luggage": avg_luggage
            }])
            pred = model.predict(input_df)
            proba = model.predict_proba(input_df)[0]
            crowd = le_crowd.inverse_transform(pred)[0]
            confidence = round(max(proba)*100, 2)

            conversation_memory["last_input"] = {
                "time": time_input,
                "passengers": passengers_input,
                "weather": weather_encoded,
                "predicted_crowd": crowd
            }

            action = "Normal operations"
            if crowd == "High":
                action = "Open additional counters"
            elif crowd == "Medium":
                action = "Monitor passenger flow"

            text = f"Based on current passenger trends, {airport_key} is expected to experience {crowd.lower()} congestion around {time_input}:00 (confidence: {confidence}%)."

            if "explain" in intents:
                text += f" This is mainly due to the current passenger load of approximately {passengers_input} travelers."

            text += f" | Action: {action}"
            response.append(text)

        # =========================
        # DELAY PREDICTION
        # =========================
        if "delay" in intents:
            input_df = pd.DataFrame([{
                "time": time_input,
                "passengers": passengers_input,
                "day_type": day_encoded,
                "weather": weather_encoded,
                "event_day": event_day,
                "flight_count": flight_count,
                "avg_luggage": avg_luggage
            }])
            pred = delay_model.predict(input_df)
            delay = le_delay.inverse_transform(pred)[0]

            if (adaptive_state["global"]["delay_bias"] + adaptive_state["airports"].get(airport_key, {}).get("delay_bias", 0)) > 0 and delay == "Medium":
                delay = "High"
            elif (adaptive_state["global"]["delay_bias"] + adaptive_state["airports"].get(airport_key, {}).get("delay_bias", 0)) < 0 and delay == "High":
                delay = "Medium"

            action = "No delay actions needed"
            if delay == "High":
                action = "Hold departures"
            elif delay == "Medium":
                action = "Optimize boarding"

            text = f"Flights at {airport_key} are expected to have {delay.lower()} delays based on current traffic and operational conditions."

            if "explain" in intents:
                text += f" This is influenced by the current traffic load of around {passengers_input} passengers."

            text += f" | Action: {action}"
            response.append(text)

        # =========================
        # RISK ANALYSIS
        # =========================
        if "risk" in intents:
            avg_passengers = local_df["passengers"].mean()

            risk = "Low"
            if avg_passengers > 1400:
                risk = "High"
            elif avg_passengers > 900:
                risk = "Medium"

            text = f"Operational risk at {airport_key} is currently assessed as {risk.lower()} based on passenger traffic and environmental conditions."

            if "explain" in intents:
                text += f" (avg passengers: {int(avg_passengers)})"

            response.append(text)

        # =========================
        # PEAK ANALYSIS
        # =========================
        if "peak" in intents:
            peak = int(local_df.groupby("time")["passengers"].mean().idxmax())
            response.append(f"Peak traffic at {airport_key} is expected around {peak}:00 based on historical passenger patterns.")

        if not response:
            response.append("No relevant information found")

        learning_note = ""
        if adaptive_state["global"]["passenger_bias"] > 0 or adaptive_state["global"]["delay_bias"] > 0:
            learning_note = " [Adaptive tuning active]"

        # =========================
        # GPT-STYLE FINAL RESPONSE
        # =========================
        final_response = " ".join(response)

        conversational_reply = f"{final_response}"

        # Add soft follow-up
        conversational_reply += " Let me know if you'd also like insights on delays, risks, or peak traffic."

        return jsonify({
            "reply": conversational_reply + learning_note
        })

    except Exception as e:
        return jsonify({
            "reply": f"System error: {str(e)}"
        })

def get_metar(icao):
    url = f"https://aviationweather.gov/api/data/metar?ids={icao}&format=json"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if len(data) > 0:
            return data[0]
    
    return None

def parse_metar(data):
    weather = data.get("wxString")
    if not weather:
        weather = data.get("rawOb", "Clear")
    vis = data.get("visib", "10")

    try:
        if isinstance(vis, str):
            vis = vis.replace("SM", "")  # remove unit
            if "+" in vis:
                vis = vis.replace("+", "")
            if "/" in vis:
                num, den = vis.split("/")
                vis = float(num) / float(den)
            visibility = min(float(vis) * 1609, 10000)
        else:
            visibility = min(float(vis) * 1609, 10000)
    except:
        visibility = 5000
    temp = data.get("temp")
    if temp is None:
        temp = data.get("tempC", 25)
    wind = data.get("wspd", 5)

    return weather, visibility, temp, wind

# Cache to avoid repeated API calls
# Cache to avoid repeated API calls
weather_cache = {}

# =========================
# OpenWeather API integration
# =========================
def get_real_weather(city):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        data = response.json()

        if response.status_code != 200:
            raise Exception("API error")

        weather = data["weather"][0]["main"]
        visibility = data.get("visibility", 5000)
        temp = data["main"]["temp"]
        wind = data["wind"]["speed"] * 3.6  # convert m/s → km/h

        return weather, visibility, wind, temp

    except:
        return None

# =========================
# OPENSKY CACHE (FLIGHTS)
# =========================
flights_cache = {}
FLIGHTS_CACHE_TTL = 10  # seconds

def get_city_weather(city):
    try:
        # Try real API first
        real = get_real_weather(city)
        if real:
            return real

        # Fallback to wttr.in
        if city in weather_cache:
            return weather_cache[city]

        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=3)
        data = response.json()

        current = data["current_condition"][0]
        weather_desc = current["weatherDesc"][0]["value"]

        visibility_km = float(current.get("visibility", 5))
        visibility = int(visibility_km * 1000)

        wind = int(current.get("windspeedKmph", 5))
        temp = int(current.get("temp_C", 25))

        result = (weather_desc, visibility, wind, temp)
        weather_cache[city] = result

        return result

    except:
        return "Haze", 4000, 5, 25

AIRPORTS = {
    "Delhi": {"icao": "VIDP", "city": "Delhi"},
    "Mumbai": {"icao": "VABB", "city": "Mumbai"},
    "Dubai": {"icao": "OMDB", "city": "Dubai"},
    "Singapore": {"icao": "WSSS", "city": "Singapore"},
    "London": {"icao": "EGLL", "city": "London"},
    "Heathrow": {"icao": "EGLL", "city": "London"},
    "Frankfurt": {"icao": "EDDF", "city": "Frankfurt"},
    "JFK": {"icao": "KJFK", "city": "New York"},
    "LAX": {"icao": "KLAX", "city": "Los Angeles"},
    "Sydney": {"icao": "YSSY", "city": "Sydney"},
    "Johannesburg": {"icao": "FAOR", "city": "Johannesburg"}
}

# =========================
# AIRPORT KNOWLEDGE BASE
# =========================
AIRPORT_INFO = {
    "Delhi": {
        "full_name": "Indira Gandhi International Airport",
        "country": "India",
        "city": "Delhi",
        "iata": "DEL",
        "icao": "VIDP",
        "terminals": 3,
        "runways": 4,
        "type": "International",
        "annual_passengers": "75+ million",
        "famous_for": "One of the busiest airports in Asia"
    },

    "Dubai": {
        "full_name": "Dubai International Airport",
        "country": "United Arab Emirates",
        "city": "Dubai",
        "iata": "DXB",
        "icao": "OMDB",
        "terminals": 3,
        "runways": 2,
        "type": "International",
        "annual_passengers": "85+ million",
        "famous_for": "Global hub for Emirates Airlines"
    },

    "Singapore": {
        "full_name": "Singapore Changi Airport",
        "country": "Singapore",
        "city": "Singapore",
        "iata": "SIN",
        "icao": "WSSS",
        "terminals": 4,
        "runways": 2,
        "type": "International",
        "annual_passengers": "68+ million",
        "famous_for": "World-class passenger experience"
    },

    "Heathrow": {
        "full_name": "London Heathrow Airport",
        "country": "United Kingdom",
        "city": "London",
        "iata": "LHR",
        "icao": "EGLL",
        "terminals": 4,
        "runways": 2,
        "type": "International",
        "annual_passengers": "79+ million",
        "famous_for": "Europe's busiest international airport"
    },

    "Frankfurt": {
        "full_name": "Frankfurt Airport",
        "country": "Germany",
        "city": "Frankfurt",
        "iata": "FRA",
        "icao": "EDDF",
        "terminals": 2,
        "runways": 4,
        "type": "International",
        "annual_passengers": "70+ million",
        "famous_for": "Major European cargo and transit hub"
    },

    "JFK": {
        "full_name": "John F. Kennedy International Airport",
        "country": "United States",
        "city": "New York",
        "iata": "JFK",
        "icao": "KJFK",
        "terminals": 6,
        "runways": 4,
        "type": "International",
        "annual_passengers": "60+ million",
        "famous_for": "Major gateway to the United States"
    },

    "LAX": {
        "full_name": "Los Angeles International Airport",
        "country": "United States",
        "city": "Los Angeles",
        "iata": "LAX",
        "icao": "KLAX",
        "terminals": 9,
        "runways": 4,
        "type": "International",
        "annual_passengers": "75+ million",
        "famous_for": "Major Pacific and Hollywood aviation hub"
    },

    "Sydney": {
        "full_name": "Sydney Kingsford Smith Airport",
        "country": "Australia",
        "city": "Sydney",
        "iata": "SYD",
        "icao": "YSSY",
        "terminals": 3,
        "runways": 3,
        "type": "International",
        "annual_passengers": "44+ million",
        "famous_for": "Australia's busiest airport"
    },

    "Johannesburg": {
        "full_name": "O. R. Tambo International Airport",
        "country": "South Africa",
        "city": "Johannesburg",
        "iata": "JNB",
        "icao": "FAOR",
        "terminals": 6,
        "runways": 2,
        "type": "International",
        "annual_passengers": "21+ million",
        "famous_for": "Africa's busiest aviation hub"
    }
}

# =========================
# LOAD AIRPORT JSON (COORDINATES)
# =========================

try:
    with open("airports.json", "r") as f:
        airport_coords = json.load(f)
except:
    airport_coords = {}

# =========================
# FLIGHTS (OPENSKY) ENDPOINT
# =========================
@app.route("/flights", methods=["GET"])
def get_flights():
    airport = request.args.get("airport")

    if not airport:
        return jsonify({"flights": []})

    airport_key = airport.strip()
    # Try exact match
    coords = airport_coords.get(airport_key)

    # If not found, try partial match (IMPORTANT FIX)
    if not coords:
        for key in airport_coords:
            if airport_key.lower() in key.lower():
                coords = airport_coords[key]
                break

    if not coords:
        return jsonify({"flights": []})

    lat = coords["lat"]
    lon = coords["lon"]

    cache_key = f"{airport_key}"

    # Check cache
    if cache_key in flights_cache:
        cached_time, cached_data = flights_cache[cache_key]
        if time.time() - cached_time < FLIGHTS_CACHE_TTL:
            return jsonify({"flights": cached_data})

    try:
        url = f"https://opensky-network.org/api/states/all?lamin={lat-1}&lomin={lon-1}&lamax={lat+1}&lomax={lon+1}"
        response = requests.get(url, timeout=5)
        data = response.json()

        flights = []

        for f in data.get("states", []):
            if f[5] is None or f[6] is None:
                continue

            flights.append({
                "lat": f[6],
                "lon": f[5],
                "callsign": f[1].strip() if f[1] else "FLIGHT",
                "velocity": f[9],
                "altitude": f[7]
            })

        # If no flights from OpenSky → simulate minimal traffic
        if len(flights) == 0:
            import random

            for i in range(5):
                flights.append({
                    "lat": lat + random.uniform(-0.5, 0.5),
                    "lon": lon + random.uniform(-0.5, 0.5),
                    "callsign": f"SIM{i}",
                    "velocity": random.randint(200, 800),
                    "altitude": random.randint(2000, 12000)
                })

        # Save to cache
        flights_cache[cache_key] = (time.time(), flights)

        return jsonify({"flights": flights})

    except Exception as e:
        return jsonify({"flights": [], "error": str(e)})

# =========================
# Dashboard Data Endpoint
@app.route("/dashboard-data", methods=["GET"])
def dashboard_data():
    state = request.args.get("state")
    airport = request.args.get("airport")
    # =========================
    # UNIFY AIRPORT KEY
    # =========================
    airport_key = airport.strip() if airport else None
    coords = airport_coords.get(airport_key, None)

    # Default fallback
    if not airport:
        filtered_df = df.copy()
    else:
        filename = airport_key.replace(" ", "_") + ".csv"
        filepath = os.path.join("airport_data", filename)

        if os.path.exists(filepath):
            filtered_df = pd.read_csv(filepath)
        else:
            filtered_df = df.copy()

    # Graph 1
    time_counts = filtered_df.groupby("time")["passengers"].mean().to_dict()

    # Graph 2
    crowd_counts = filtered_df["crowd"].value_counts().to_dict()

    # Graph 3
    day_counts = filtered_df.groupby("day_type")["passengers"].mean().to_dict()

    # Default METAR values
    delay = "Low"
    weather = "Clear"
    visibility = 10
    temp = 25

    # METAR integration
    icao = AIRPORTS.get(airport_key, {}).get("icao")

    # =========================
    # HYBRID WEATHER (NO CLEAR DEFAULT)
    # =========================
    try:
        if icao:
            metar_data = get_metar(icao)
            if metar_data:
                weather, visibility, temp, wind = parse_metar(metar_data)
            else:
                raise Exception("METAR unavailable")
        else:
            raise Exception("No ICAO")

    except:
        try:
            if airport:
                city = airport.replace("Airport", "").replace("International", "").strip()
                city = city.split()[0] if city else "Delhi"
            else:
                city = "Delhi"

            city = AIRPORTS.get(airport_key, {}).get("city", city)

            # ALWAYS fetch real city weather (NO 'Clear' fallback)
            weather, visibility, wind, temp = get_city_weather(city)

        except:
            # Last fallback (rare case only)
            weather = "Haze"
            visibility = 4000
            wind = 5
            temp = 25

    # Ensure wind has a default
    try:
        wind
    except:
        wind = 5
    # =========================
    # ATC RISK SCORING SYSTEM
    # =========================

    weather_lower = str(weather).lower()
    risk_score = 0

    # Visibility impact (stronger)
    if visibility < 1000:
        risk_score += 50
    elif visibility < 4000:
        risk_score += 30

    # Weather impact (stronger)
    if "fog" in weather_lower:
        risk_score += 50
    elif "haze" in weather_lower or "hz" in weather_lower:
        risk_score += 25
        visibility = min(visibility, 5000)
    elif "storm" in weather_lower or "thunder" in weather_lower:
        risk_score += 60
    elif "rain" in weather_lower:
        risk_score += 30
        visibility = min(visibility, 6000)

    # Wind impact
    if wind > 40:
        risk_score += 40
    elif wind > 25:
        risk_score += 20

    # Traffic impact (from dataset)
    avg_passengers = filtered_df["passengers"].mean()

    if avg_passengers > 1400:
        risk_score += 40
    elif avg_passengers > 900:
        risk_score += 25

# =========================
# ML DELAY PREDICTION (REAL-TIME)
# =========================
    try:
        from datetime import datetime

        current_hour = datetime.now().hour

        # Weather encoding using LabelEncoder as in training
        try:
            weather_clean = normalize_weather(weather)
            weather_encoded = le_weather.transform([weather_clean])[0]
        except:
            weather_encoded = 0

        # Estimate flights dynamically
        try:
            BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:5000")
            flights_res = requests.get(f"{BASE_URL}/flights?airport={airport_key}")
            flights_data = flights_res.json()
            flight_count = len(flights_data.get("flights", []))
        except:
            flight_count = int(avg_passengers / 50)

        sample = [[
            current_hour,
            avg_passengers,
            0,
            weather_encoded,
            0,
            flight_count,
            2
        ]]

        pred_delay = delay_model.predict(sample)
        delay = le_delay.inverse_transform(pred_delay)[0]

    except Exception as e:
        delay = "Low"

# =========================
# Combine ML + Risk (FINAL DECISION)
# =========================
    # Enhanced Delay Decision (aligned with frontend)
    if risk_score >= 80 or (avg_passengers > 1400 and weather_lower != "clear"):
        delay = "High"
    elif risk_score >= 50 or avg_passengers > 1000:
        if delay != "High":
            delay = "Medium"

# =========================
# ATC ALERT SYSTEM
# =========================
    alerts = []

    if visibility < 1000:
        alerts.append("⚠️ Low Visibility - Landing Risk")

    if "fog" in weather_lower:
        alerts.append("🌫 Fog Conditions - Delay Expected")

    if wind > 40:
        alerts.append("💨 High Wind - Flight Instability")

    if avg_passengers > 1500:
        alerts.append("👥 Heavy Traffic - Congestion Risk")

    if risk_score >= 80:
        alerts.append("🚨 CRITICAL: ATC Intervention Required")

# =========================
# DECISION ACTION ENGINE
# =========================
    actions = []

    if delay == "High":
        actions.append("Delay departures")
        actions.append("Hold incoming aircraft")
        actions.append("Deploy emergency ground staff")
    elif delay == "Medium":
        actions.append("Open extra security lanes")
        actions.append("Increase boarding efficiency")
    else:
        actions.append("Normal operations")

    return jsonify({
        "time_vs_passengers": time_counts,
        "crowd_distribution": crowd_counts,
        "day_vs_passengers": day_counts,
        "weather": weather,
        "visibility": visibility,
        "temperature": temp,
        "delay": delay,
        "risk_score": risk_score,
        "alerts": alerts,
        "coordinates": coords
        ,"actions": actions
    })

# =========================
# RUN SERVER
# =========================


# =========================
# ADVANCED FEEDBACK ENDPOINT
# =========================
@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    correct = data.get("correct")
    issue = data.get("issue", "")
    airport = data.get("airport", "global")

    # Ensure airport entry exists
    if airport not in adaptive_state["airports"]:
        adaptive_state["airports"][airport] = {"passenger_bias": 0, "delay_bias": 0}

    weight = 2 if not correct else -1  # weighted learning

    if "crowd" in issue:
        adaptive_state["global"]["passenger_bias"] += 10 * weight
        adaptive_state["airports"][airport]["passenger_bias"] += 20 * weight

    if "delay" in issue:
        adaptive_state["global"]["delay_bias"] += 1 * weight
        adaptive_state["airports"][airport]["delay_bias"] += 2 * weight

    # Save history (recent more important)
    adaptive_state["history"].append({
        "correct": correct,
        "issue": issue,
        "airport": airport
    })

    # Keep only last 50 feedbacks (recency importance)
    adaptive_state["history"] = adaptive_state["history"][-50:]

    # =========================
    # STORE FEEDBACK AS TRAINING DATA
    # =========================
    try:
        # Intelligent label correction based on last prediction
        last = conversation_memory.get("last_input", {})
        predicted = last.get("predicted_crowd", "Medium")

        if not correct:
            if predicted == "High":
                corrected = "Low"
            elif predicted == "Low":
                corrected = "High"
            else:
                corrected = "High"
        else:
            corrected = predicted

        try:
            crowd_val = le_crowd.transform([corrected])[0]
        except:
            crowd_val = le_crowd.transform(["Medium"])[0]

        # Safe encoding for delay
        try:
            delay_val = le_delay.transform([data.get("correct_delay", "Medium")])[0]
        except:
            delay_val = le_delay.transform(["Medium"])[0]

        # Safe weather encoding
        try:
            weather_clean = normalize_weather(data.get("weather", "Clear"))
            weather_val = le_weather.transform([weather_clean])[0]
        except:
            weather_val = 0

        # Use real last input context (intelligent learning)
        last = conversation_memory.get("last_input", {})

        time_val = last.get("time", data.get("time", 12))
        passengers_val = last.get("passengers", data.get("passengers", 1000))

        import random
        sample = {
            "time": time_val,
            "passengers": passengers_val,
            "day_type": le_day.transform(["Weekday"])[0],
            "weather": weather_val,
            "event_day": 0,
            "flight_count": int(passengers_val/50),
            "avg_luggage": random.randint(1,4),
            "crowd": crowd_val,
            "delay_label": delay_val
        }

        global feedback_df
        feedback_df = pd.concat([feedback_df, pd.DataFrame([sample])], ignore_index=True)

        print("Saving feedback to:", FEEDBACK_DATA_FILE)
        print("Current directory:", os.getcwd())

        feedback_df.to_csv(FEEDBACK_DATA_FILE, index=False)

        print("Feedback sample added:", sample)

    except Exception as e:
        print("FEEDBACK ERROR:", e)

    # SAVE TO FILE (PERSISTENCE)
    with open(ADAPTIVE_FILE, "wb") as f:
        pkl.dump(adaptive_state, f)

    # =========================
    # AUTO-RETRAIN TRIGGER (OPTIONAL)
    # =========================
    if len(feedback_df) % 10 == 0 and len(feedback_df) > 0:
        try:
            combined_df = pd.concat([df, feedback_df], ignore_index=True)

            X_combined = combined_df[[
                "time","passengers","day_type","weather",
                "event_day","flight_count","avg_luggage"
            ]]

            model.fit(X_combined, combined_df["crowd"])
            delay_model.fit(X_combined, combined_df["delay_label"])
        except Exception as e:
            print("AUTO RETRAIN ERROR:", e)

    return jsonify({
        "status": "adaptive learning updated",
        "global": adaptive_state["global"],
        "airport": adaptive_state["airports"][airport]
    })


# =========================
# RETRAIN FUNCTION (ML)
# =========================
@app.route("/retrain", methods=["POST"])
def retrain_model():
    global model, delay_model

    try:
        if len(feedback_df) < 10:
            return jsonify({"status": "Not enough data to retrain"})

        # =========================
        # MERGE ORIGINAL + FEEDBACK DATA
        # =========================
        combined_df = pd.concat([df, feedback_df], ignore_index=True)

        X_combined = combined_df[[
            "time","passengers","day_type","weather",
            "event_day","flight_count","avg_luggage"
        ]]

        model.fit(X_combined, combined_df["crowd"])
        delay_model.fit(X_combined, combined_df["delay_label"])

        # Save updated models
        pickle.dump(model, open("crowd_model.pkl", "wb"))
        pickle.dump(delay_model, open("delay_model.pkl", "wb"))

        return jsonify({"status": "Model retrained successfully"})

    except Exception as e:
        return jsonify({"status": f"Retrain failed: {str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
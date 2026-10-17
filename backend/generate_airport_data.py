import pandas as pd
import numpy as np
import os

# Create folder
os.makedirs("airport_data", exist_ok=True)

# Global Hub Airports (continental focus)
airports = [
    "Delhi",
    "Mumbai",
    "Dubai",
    "Singapore",
    "London",
    "Frankfurt",
    "JFK",
    "LAX",
    "Sydney",
    "Johannesburg"
]

# Function to generate data
def generate_data(airport_name):
    n = 5000

    time = np.random.randint(0, 24, n)
    day_type = np.random.choice(["Weekday", "Weekend"], n)
    weather = np.random.choice(["Clear", "Rain", "Fog"], n)
    event_day = np.random.choice([0, 1], n)

    passengers = []
    flight_count = []
    avg_luggage = []

    for t, d, w, e in zip(time, day_type, weather, event_day):

        # Base passengers
        if 6 <= t <= 10 or 17 <= t <= 21:
            p = np.random.randint(1000, 2000)
        else:
            p = np.random.randint(200, 900)

        # Weekend effect
        if d == "Weekend":
            p += 300

        # Event impact
        if e == 1:
            p += 400

        # Weather impact
        if w == "Rain":
            p -= 200
        elif w == "Fog":
            p -= 300

        # Airport uniqueness
        factor = sum(ord(c) for c in airport_name) % 200
        p += factor

        p = max(100, p)
        passengers.append(p)

        # Flights proportional to passengers
        flight_count.append(int(p / 50))

        # Random luggage count
        avg_luggage.append(np.random.randint(1, 5))

    # Crowd classification
    crowd = []
    for p in passengers:
        if p > 1500:
            crowd.append("High")
        elif p > 700:
            crowd.append("Medium")
        else:
            crowd.append("Low")

    df = pd.DataFrame({
        "time": time,
        "day_type": day_type,
        "weather": weather,
        "event_day": event_day,
        "passengers": passengers,
        "flight_count": flight_count,
        "avg_luggage": avg_luggage,
        "crowd": crowd
    })

    return df


# Generate CSV for all airports
for airport in airports:
    df = generate_data(airport)

    airport_key = airport.strip() if airport else None
    filename = airport_key.replace(" ", "_") + ".csv"

    df.to_csv(f"airport_data/{filename}", index=False)

print("✅ All airport CSV files generated!")
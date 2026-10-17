# ✈️ Airsys — Airport AI System

🌐 **Live demo:** [airsys.netlify.app](https://airsys.netlify.app)

An AI-powered airport intelligence platform that predicts **crowd levels**, **flight delays** and **operational risk** for major international airports. It combines machine-learning models with live weather and flight data, and includes a chat assistant that answers questions about airports in plain English.

## Features

- **Crowd prediction**: a Random Forest model classifies expected congestion (Low / Medium / High) from time of day, day type, weather, events and flight count
- **Delay prediction**: a second Random Forest model estimates flight delay levels
- **Risk assessment**: operational risk based on passenger traffic and environmental conditions
- **AI chat assistant**: ask things like *"How crowded is Dubai at 6 pm?"* or *"What's the IATA code for Heathrow?"*
- **Live dashboard**: radar, weather, risk and traffic panels with charts
- **Live data**: weather from METAR (aviationweather.gov) and OpenWeather / wttr.in, nearby flights from the OpenSky Network
- **Adaptive learning**: user feedback is stored and the models retrain periodically to correct their bias

**Airports covered:** Delhi, Mumbai, Dubai, Singapore, London Heathrow, Frankfurt, New York JFK, Los Angeles LAX, Sydney, Johannesburg

## Tech Stack

| Layer | Technologies |
|---|---|
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Backend | Python, Flask, Flask-CORS, Gunicorn |
| ML | scikit-learn (RandomForestClassifier), pandas, NumPy |
| Deployment | Netlify (frontend), Render (backend) |

## Project Structure

```
Airsys/
├── backend/                     → deployed on Render
│   ├── dataset.py               Flask API + model training/prediction
│   ├── generate_airport_data.py Generates per-airport datasets
│   ├── airport_data/            Per-airport CSV datasets
│   ├── *.pkl                    Trained models and label encoders
│   ├── airports.json            Airport coordinates
│   ├── feedback_data.csv        User feedback for adaptive learning
│   ├── requirements.txt
│   └── Procfile
└── frontend/                    → deployed on Netlify
    ├── index.html               Landing page
    ├── airport.html             Dashboard + AI chat
    └── *.png                    Images and logo
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Crowd prediction |
| POST | `/chat` | AI assistant chat |
| GET | `/dashboard-data?airport=<name>` | Dashboard data (weather, crowd, delay, risk) |
| GET | `/flights?airport=<name>` | Live flights near an airport |
| POST | `/feedback` | Submit feedback for adaptive learning |

## Running Locally

**Backend**

```bash
cd backend
pip install -r requirements.txt
python dataset.py          # runs on http://localhost:10000
```

Optional environment variable: `OPENWEATHER_API_KEY` (falls back to wttr.in if not set).

**Frontend**

Open `frontend/index.html` in a browser. The frontend uses the hosted backend at `https://airport-ai-backend.onrender.com`. To use a local backend, change that URL in `frontend/airport.html`.

## Deployment

- **Backend**: Render, with Root Directory set to `backend`
- **Frontend**: Netlify, by uploading the `frontend/` folder ([airsys.netlify.app](https://airsys.netlify.app))

## Author

**Utkarsh Mehta**: [GitHub](https://github.com/UtkarshMehta7)

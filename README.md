# Airsys

AI-powered airport crowd and delay prediction system.

## Structure

```
Airsys/
├── backend/    Flask API (dataset.py) + scikit-learn models + airport datasets  → deployed on Render
└── frontend/   Static website (index.html, airport.html, images)                → deployed on Netlify
```

The frontend calls the backend at `https://airport-ai-backend.onrender.com`.

## Run the backend locally

```bash
cd backend
pip install -r requirements.txt
python dataset.py
```

## Run the frontend locally

Open `frontend/index.html` in a browser.

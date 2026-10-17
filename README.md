# Airsys

AI-powered airport crowd and delay prediction system.

## Structure

- **Backend (repo root)**: Flask API (`dataset.py`) with scikit-learn models (`*.pkl`) and airport datasets (`airport_data/`). Deployed on **Render** via `Procfile`.
- **`frontend/`**: static website (`index.html`, `airport.html`, images). Deployed on **Netlify**. It calls the Render backend at `https://airport-ai-backend.onrender.com`.

## Run the backend locally

```bash
pip install -r requirements.txt
python dataset.py
```

# Real-Time Face Verification API
**Python · OpenCV · DeepFace · FastAPI**

---

## What it does
- Live webcam capture via OpenCV with background multithreading (inference every 30 frames, non-blocking)
- DeepFace face matching against a stored reference image
- REST API endpoints — upload images, get JSON responses (match status, distance score, threshold)
- Graceful error handling: HTTP 422 when no face is detected instead of crashing

## Endpoints
| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/upload-reference` | Save reference face image |
| POST | `/verify` | Match face → returns verified, distance, threshold |
| POST | `/analyze` | Returns age, gender, emotion, race |
| POST | `/verify-and-analyze` | Both in one request |
| POST | `/webcam/start` | Start live webcam + background inference thread |
| POST | `/webcam/stop` | Stop webcam thread |
| GET | `/webcam/status` | Latest result from live webcam thread |

Interactive docs at **`http://127.0.0.1:8000/docs`**

---

## Setup & Run

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
uvicorn main:app --reload
```

---

## Push to GitHub

```bash
# ── One-time global config ──────────────────────────────────────────
git config --global user.name "Your Name"
git config --global user.email "you@example.com"

# ── Inside your project folder ──────────────────────────────────────
git init
git add .
git commit -m "feat: Real-Time Face Verification API with FastAPI"

# Create a NEW empty repo on github.com (no README, no .gitignore)
# Then copy its URL and run:

git remote add origin https://github.com/YOUR_USERNAME/face-recognition-api.git
git branch -M main
git push -u origin main

# ── Every update after this ─────────────────────────────────────────
git add .
git commit -m "describe your change"
git push
```

> **Note:** `reference_images/` is gitignored — your reference face photo will never be pushed.

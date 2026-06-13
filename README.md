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
## Key Learning Outcomes

- Built REST APIs using FastAPI
- Implemented face verification with DeepFace
- Used OpenCV for real-time webcam processing
- Implemented multithreading for non-blocking inference
- Added exception handling and API validation

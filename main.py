import threading
import cv2
import numpy as np
from deepface import DeepFace
from fastapi import FastAPI, File, UploadFile, HTTPException
from pathlib import Path

app = FastAPI(title="Real-Time Face Verification API", version="2.0")

REF_DIR = Path("reference_images")
REF_DIR.mkdir(exist_ok=True)
REF_PATH = REF_DIR / "reference.jpg"

# ── Shared webcam state (thread-safe) ─────────────────────────────────────────
_lock = threading.Lock()
_webcam_running = False
_cap: cv2.VideoCapture | None = None
_cam_state: dict = {
    "face_match": False,
    "distance": None,
    "threshold": None,
    "error": None,
}


# ── Background inference thread (runs every 30 frames, non-blocking) ──────────
def _run_inference(frame: np.ndarray) -> None:
    try:
        r = DeepFace.verify(frame, str(REF_PATH), enforce_detection=True)
        with _lock:
            _cam_state.update(
                face_match=r["verified"],
                distance=round(r["distance"], 4),
                threshold=round(r["threshold"], 4),
                error=None,
            )
    except ValueError:
        with _lock:
            _cam_state.update(face_match=False, error="No face detected in frame")


def _webcam_loop() -> None:
    global _cap, _webcam_running
    _cap = cv2.VideoCapture(0)
    _cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    count = 0
    while _webcam_running:
        ret, frame = _cap.read()
        if not ret:
            break
        if count % 30 == 0 and REF_PATH.exists():
            threading.Thread(
                target=_run_inference,
                args=(frame.copy(),),
                daemon=True,
            ).start()
        count += 1
    _cap.release()
    _webcam_running = False


# ── Image decode helper ────────────────────────────────────────────────────────
async def _decode(file: UploadFile) -> np.ndarray:
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image file")
    return img


# ── REST Endpoints ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}


@app.post("/upload-reference")
async def upload_reference(file: UploadFile = File(...)):
    """Save the reference face image to compare against."""
    img = await _decode(file)
    cv2.imwrite(str(REF_PATH), img)
    return {"message": "Reference image saved"}


@app.post("/verify")
async def verify(file: UploadFile = File(...)):
    """
    Accepts an image upload, decodes it with OpenCV, runs DeepFace verification
    against the stored reference image. Returns match status, distance score,
    and verification threshold. Returns HTTP 422 if no face is detected.
    """
    if not REF_PATH.exists():
        raise HTTPException(404, "No reference image. POST to /upload-reference first.")
    img = await _decode(file)
    try:
        r = DeepFace.verify(img, str(REF_PATH), enforce_detection=True)
        return {
            "verified": r["verified"],
            "distance": round(r["distance"], 4),
            "threshold": round(r["threshold"], 4),
            "model": r["model"],
        }
    except ValueError as e:
        raise HTTPException(422, f"No face detected: {e}")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Return age, gender, dominant emotion, and race for a face."""
    img = await _decode(file)
    try:
        results = DeepFace.analyze(
            img,
            actions=["age", "gender", "emotion", "race"],
            enforce_detection=True,
        )
        face = results[0] if isinstance(results, list) else results
        return {
            "age": face["age"],
            "gender": face["dominant_gender"],
            "emotion": face["dominant_emotion"],
            "all_emotions": {k: round(v, 2) for k, v in face["emotion"].items()},
            "race": face["dominant_race"],
        }
    except ValueError as e:
        raise HTTPException(422, f"No face detected: {e}")


@app.post("/verify-and-analyze")
async def verify_and_analyze(file: UploadFile = File(...)):
    """Verify + analyze in a single request."""
    if not REF_PATH.exists():
        raise HTTPException(404, "No reference image. POST to /upload-reference first.")
    img = await _decode(file)
    try:
        v = DeepFace.verify(img, str(REF_PATH), enforce_detection=True)
        a = DeepFace.analyze(
            img, actions=["age", "gender", "emotion"], enforce_detection=True
        )
        face = a[0] if isinstance(a, list) else a
        return {
            "verified": v["verified"],
            "distance": round(v["distance"], 4),
            "age": face["age"],
            "gender": face["dominant_gender"],
            "emotion": face["dominant_emotion"],
        }
    except ValueError as e:
        raise HTTPException(422, f"No face detected: {e}")


# ── Webcam Control Endpoints ───────────────────────────────────────────────────
@app.post("/webcam/start")
def webcam_start():
    """
    Start live webcam capture. Spawns a background thread that reads frames
    and runs DeepFace inference every 30 frames without blocking the feed.
    """
    global _webcam_running
    if _webcam_running:
        return {"message": "Webcam already running"}
    if not REF_PATH.exists():
        raise HTTPException(404, "Upload a reference image first via /upload-reference.")
    _webcam_running = True
    threading.Thread(target=_webcam_loop, daemon=True).start()
    return {"message": "Webcam started — poll /webcam/status for results"}


@app.post("/webcam/stop")
def webcam_stop():
    """Stop the live webcam capture thread."""
    global _webcam_running
    _webcam_running = False
    return {"message": "Webcam stopping"}


@app.get("/webcam/status")
def webcam_status():
    """Get the latest face match result from the live webcam inference thread."""
    with _lock:
        return {"running": _webcam_running, **_cam_state}

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from video_processor import analyze_video
from ai_video_processor import analyze_video_ai
from hybrid_video_processor import analyze_video_hybrid
from traffic_video_processor import analyze_traffic_video
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Real-Time Video Anomaly Detection API",
    description="Detects abnormal motion patterns in uploaded video streams using OpenCV.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


@app.get("/")
def root():
    return {
        "message": "Video Anomaly Detection API is running",
        "status": "active"
    }


@app.post("/analyze-video")
async def analyze_uploaded_video(file: UploadFile = File(...)):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = analyze_video(file_path)

    return {
        "filename": file.filename,
        "result": result
    }

@app.post("/analyze-video-ai")
async def analyze_uploaded_video_ai(file: UploadFile = File(...)):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = analyze_video_ai(file_path)

    return {
        "filename": file.filename,
        "result": result
    }

@app.post("/analyze-video-hybrid")
async def analyze_uploaded_video_hybrid(file: UploadFile = File(...)):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = analyze_video_hybrid(file_path)

    return {
        "filename": file.filename,
        "result": result
    }

@app.post("/analyze-traffic-video")
async def analyze_traffic(file: UploadFile = File(...)):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = analyze_traffic_video(file_path)

    return {
        "filename": file.filename,
        "result": result
    }
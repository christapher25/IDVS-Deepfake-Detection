import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

# --- IMPORT OUR NEW QUAD-CORE ENGINE ---
from inference_logic_4 import process_video_logic

app = FastAPI(title="Apex Minds Deepfake Shield - Quad Core Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("⏳ INITIALIZING QUAD-CORE API SERVER...")

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    # 1. Save the uploaded video temporarily
    temp_input_path = f"temp_{file.filename}"
    temp_output_path = f"analyzed_{file.filename}.mp4"

    try:
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Feed it to the Quad-Core Engine
        print(f"🎬 Sending {file.filename} to Quad-Core Engine...")
        engine_result = process_video_logic(temp_input_path, temp_output_path)

        # 3. Format the data to ensure frontend compatibility
        if engine_result["status"] == "error":
            return {
                "result": "ERROR",
                "confidence": 0,
                "reason": engine_result["message"]
            }

        # The frontend usually expects a decimal for confidence (0.0 to 1.0)
        # Our engine outputs a percentage (e.g., 95.0), so we convert it:
        confidence_decimal = engine_result["confidence"] / 100.0

        # Construct a rich response containing all the new Quad-Core stats!
        return {
            "result": engine_result["verdict"],
            "confidence": confidence_decimal,
            "reason": f"Analyzed {engine_result['duration']}s of footage. Blinks detected: {engine_result['blinks']}.",
            "anomaly_triggered": engine_result["anomaly"]
        }

    finally:
        # 4. Clean up temporary files to save hard drive space
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)

if __name__ == "__main__":
    uvicorn.run("backend_api:app", host="127.0.0.1", port=8000, reload=True)
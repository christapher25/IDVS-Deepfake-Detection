import streamlit as st
import tempfile
import os
import time
from inference_logic_4 import process_video_logic, model

# --- PAGE CONFIG ---
st.set_page_config(page_title="Deepfake Detector", page_icon="🛡️", layout="wide")

st.title("🛡️ Deepfake Detection System")
st.markdown("### Upload a video to scan for AI-generated faces.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Status")
    if model is not None:
        st.success("✅ Model Loaded Successfully")
    else:
        st.error("❌ Model Failed to Load")

    st.info("Supported formats: MP4, MOV, AVI")

# --- MAIN UPLOAD ---
uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # 1. Save Uploaded File to Temp
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    tfile.close()  # CRITICAL: Close the file immediately after writing

    st.video(video_path)

    if st.button("🔍 Scan Video"):
        with st.spinner("Analyzing frames... This may take a moment."):
            # Run Inference
            output_video_path = video_path.replace(".mp4", "_processed.mp4")
            verdict, processed_video = process_video_logic(video_path, output_video_path)

            # --- RESULTS ---
            st.divider()
            if verdict == "FAKE":
                st.error(f"🚨 ALERT: Deepfake Content Detected!")
            elif verdict == "REAL":
                st.success(f"✅ VERIFIED: Real Human Detected.")
            else:
                st.warning(f"⚠️ {verdict}")

    # --- CLEANUP (The Fix) ---
    # We delay deletion slightly to ensure the OS releases the lock
    try:
        time.sleep(1)
        if os.path.exists(video_path):
            os.remove(video_path)
    except PermissionError:
        pass  # If it fails, just ignore it. Temp files get cleaned up eventually anyway.
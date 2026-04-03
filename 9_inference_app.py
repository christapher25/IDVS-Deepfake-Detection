import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import time
import math
import sys
from collections import deque
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog

# --- 1. CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIDENCE_THRESHOLD = 0.50
FACE_PADDING_RATIO = 0.50
SMOOTHING_FRAMES = 5
EAR_THRESHOLD = 0.22
CONSEC_FRAMES = 2

print(f"🚀 STARTING APEX MINDS QUAD-CORE ENGINE ON {DEVICE}...")


# --- 2. THE CUSTOM CNN BLUEPRINT ---
class DeepfakeCNN(nn.Module):
    def __init__(self):
        super(DeepfakeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# --- 3. DUAL PREPROCESSING ---
transform_standard = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

transform_imagenet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- 4. MODEL LOADER ---
def load_models():
    print("⬇️ Loading Quad-Core Arsenal...")

    cnn = DeepfakeCNN().to(DEVICE)
    if os.path.exists("models/deepfake_detector_v2.pth"):
        cnn.load_state_dict(torch.load("models/deepfake_detector_v2.pth", map_location=DEVICE, weights_only=True))
        cnn.eval()
        print("   ✅ Custom CNN Ready")
    else:
        cnn = None

    resnet = models.resnet50()
    resnet.fc = nn.Sequential(nn.Dropout(p=0.5, inplace=True), nn.Linear(resnet.fc.in_features, 1))
    if os.path.exists("models/resnet50_deepfake_v1.pth"):
        resnet.load_state_dict(torch.load("models/resnet50_deepfake_v1.pth", map_location=DEVICE, weights_only=True))
        resnet.to(DEVICE).eval()
        print("   ✅ ResNet-50 Ready")
    else:
        resnet = None

    effnet = models.efficientnet_b0()
    effnet.classifier[1] = nn.Sequential(nn.Dropout(p=0.4, inplace=True),
                                         nn.Linear(effnet.classifier[1].in_features, 1))
    if os.path.exists("models/efficientnet_deepfake_v1.pth"):
        effnet.load_state_dict(
            torch.load("models/efficientnet_deepfake_v1.pth", map_location=DEVICE, weights_only=True))
        effnet.to(DEVICE).eval()
        print("   ✅ EfficientNet-B0 Ready")
    else:
        effnet = None

    return cnn, resnet, effnet


cnn_model, resnet_model, eff_model = load_models()


# --- 5. BIOLOGY MATH ---
def compute_ear(eye_points, landmarks, w, h):
    pts = np.array([[(landmarks[p].x * w), (landmarks[p].y * h)] for p in eye_points])
    dist1 = math.hypot(pts[1][0] - pts[5][0], pts[1][1] - pts[5][1])
    dist2 = math.hypot(pts[2][0] - pts[4][0], pts[2][1] - pts[4][1])
    dist3 = math.hypot(pts[0][0] - pts[3][0], pts[0][1] - pts[3][1])
    if dist3 == 0: return 0
    return (dist1 + dist2) / (2.0 * dist3)


RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# --- 6. FILE SELECTOR UI ---
root = tk.Tk()
root.withdraw()  # Hide the ugly background window
print("\n📂 Waiting for user to select a video file...")
video_path = filedialog.askopenfilename(
    title="Apex Minds: Select Video for Analysis",
    filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv")]
)

if not video_path:
    print("❌ No video selected. Exiting System.")
    sys.exit()

print(f"🎬 Analyzing: {os.path.basename(video_path)}")

# --- 7. MAIN UI LOOP ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(video_path)

# Calculate optimal playback speed so it doesn't fast-forward
fps = cap.get(cv2.CAP_PROP_FPS)
playback_delay = int(1000 / fps) if fps > 0 else 30

# Tracking Variables
score_queue = deque(maxlen=SMOOTHING_FRAMES)
blink_counter = 0
eye_closed_frames = 0
total_frames = 0

print("\n🎥 DASHBOARD ACTIVE. Press 'q' to stop the video.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("✅ End of video reached.")
        break

    total_frames += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    ih, iw, _ = frame.shape
    current_time_sec = total_frames / fps if fps > 0 else 0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # 1. BIOLOGY EXPERT (Blinks)
            left_ear = compute_ear(LEFT_EYE, face_landmarks.landmark, iw, ih)
            right_ear = compute_ear(RIGHT_EYE, face_landmarks.landmark, iw, ih)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                eye_closed_frames += 1
            else:
                if eye_closed_frames >= CONSEC_FRAMES:
                    blink_counter += 1
                eye_closed_frames = 0

            # 2. BOUNDING BOX
            landmarks_np = np.array([[int(l.x * iw), int(l.y * ih)] for l in face_landmarks.landmark])
            x_min, y_min = np.min(landmarks_np, axis=0)
            x_max, y_max = np.max(landmarks_np, axis=0)

            pad_w, pad_h = int((x_max - x_min) * FACE_PADDING_RATIO), int((y_max - y_min) * FACE_PADDING_RATIO)
            x_min, y_min = max(0, x_min - pad_w), max(0, y_min - pad_h)
            x_max, y_max = min(iw, x_max + pad_w), min(ih, y_max + pad_h)

            try:
                face_crop = rgb_frame[y_min:y_max, x_min:x_max]
                if face_crop.size == 0: continue
                pil_face = Image.fromarray(face_crop)

                # 3. TRI-CNN INFERENCE
                t_imgnet = transform_imagenet(pil_face).unsqueeze(0).to(DEVICE)
                t_std = transform_standard(pil_face).unsqueeze(0).to(DEVICE)

                eff_p, res_p, cnn_p = 0.0, 0.0, 0.0
                active_models = 0
                total_score = 0.0

                with torch.no_grad():
                    if eff_model:
                        eff_p = torch.sigmoid(eff_model(t_imgnet)).item()
                        total_score += eff_p;
                        active_models += 1
                    if resnet_model:
                        res_p = torch.sigmoid(resnet_model(t_imgnet)).item()
                        total_score += res_p;
                        active_models += 1
                    if cnn_model:
                        cnn_p = torch.sigmoid(cnn_model(t_std)).item()
                        total_score += cnn_p;
                        active_models += 1

                raw_score = total_score / active_models if active_models > 0 else 0.5
                score_queue.append(raw_score)
                final_score = sum(score_queue) / len(score_queue)

                # 4. LIVE BIOLOGICAL OVERRIDE
                anomaly_triggered = False
                if current_time_sec > 3.0 and blink_counter <= 1:
                    final_score = min(0.99, final_score + 0.60)
                    anomaly_triggered = True

                # 5. RENDER UI GRAPHICS
                is_fake = final_score > CONFIDENCE_THRESHOLD
                confidence = final_score if is_fake else (1.0 - final_score)
                color = (0, 0, 255) if is_fake else (0, 255, 0)
                label = "FAKE" if is_fake else "REAL"

                # Draw Box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                # Draw Main Verdict
                cv2.putText(frame, f"{label} ({confidence * 100:.0f}%)", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, color, 2)

                # Draw Blink Counter & Time
                cv2.putText(frame, f"Blinks: {blink_counter} | Time: {current_time_sec:.1f}s", (x_min, y_max + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Draw Debug Stats (C=CNN, R=ResNet, E=EffNet)
                stats = f"C:{cnn_p:.2f} R:{res_p:.2f} E:{eff_p:.2f}"
                cv2.putText(frame, stats, (x_min, y_max + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Flash Anomaly Warning!
                if anomaly_triggered:
                    cv2.putText(frame, "BIOLOGICAL ANOMALY", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

            except Exception as e:
                pass

    cv2.imshow('Apex Minds Integrated Deepfake Shield', frame)

    # Sync to original video FPS so it plays naturally
    if cv2.waitKey(playback_delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import time
from collections import deque
import math

# --- FAIL-SAFE MEDIAPIPE IMPORT ---
import mediapipe as mp

try:
    mp_face_mesh = mp.solutions.face_mesh
except AttributeError:
    import mediapipe.python.solutions.face_mesh as mp_face_mesh

# ====================================================================
# --- 1. THE CONTROL PANEL (Quad-Core Edition) ---
# ====================================================================
EFFICIENTNET_PATH = "models/efficientnet_deepfake_v1.pth"
RESNET_PATH = "models/resnet50_deepfake_v1.pth"
CUSTOM_CNN_PATH = "models/deepfake_detector_v2.pth"

CONFIDENCE_THRESHOLD = 0.50
FACE_PADDING_RATIO = 0.50
SMOOTHING_FRAMES = 5
SAVE_DEBUG_CROPS = False

# --- BLINK DETECTION SETTINGS ---
EAR_THRESHOLD = 0.22
CONSEC_FRAMES = 2


# ====================================================================
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


# --- 3. HARDWARE & DUAL-PREPROCESSING ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"🚀 SYSTEM UNLEASHED: Running on {torch.cuda.get_device_name(0)} (CUDA)")
else:
    DEVICE = torch.device("cpu")
    torch.set_num_threads(4)
    print(f"⚙️ System Initialized on: CPU (CUDA not detected)")

preprocess_imagenet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

preprocess_standard = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# --- 4. LOAD ALL 3 BRAINS ---
def load_efficientnet():
    print(f"⏳ Loading EfficientNet-B0 to {DEVICE}...")
    model = models.efficientnet_b0()
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(model.classifier[1].in_features, 1)
    )
    if os.path.exists(EFFICIENTNET_PATH):
        model.load_state_dict(torch.load(EFFICIENTNET_PATH, map_location=DEVICE, weights_only=True))
        model.to(DEVICE).eval()
        return model
    return None


def load_resnet():
    print(f"⏳ Loading ResNet-50 to {DEVICE}...")
    model = models.resnet50()
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(model.fc.in_features, 1)
    )
    if os.path.exists(RESNET_PATH):
        model.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE, weights_only=True))
        model.to(DEVICE).eval()
        return model
    return None


def load_custom_cnn():
    print(f"⏳ Loading Custom DeepfakeCNN to {DEVICE}...")
    model = DeepfakeCNN()
    if os.path.exists(CUSTOM_CNN_PATH):
        model.load_state_dict(torch.load(CUSTOM_CNN_PATH, map_location=DEVICE, weights_only=True))
        model.to(DEVICE).eval()
        return model
    return None


efficientnet_model = load_efficientnet()
resnet_model = load_resnet()
custom_model = load_custom_cnn()

# --- 5. INITIALIZE BIOLOGY ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)


def compute_ear(eye_points, landmarks, w, h):
    pts = np.array([[(landmarks[p].x * w), (landmarks[p].y * h)] for p in eye_points])
    dist1 = math.hypot(pts[1][0] - pts[5][0], pts[1][1] - pts[5][1])
    dist2 = math.hypot(pts[2][0] - pts[4][0], pts[2][1] - pts[4][1])
    dist3 = math.hypot(pts[0][0] - pts[3][0], pts[0][1] - pts[3][1])
    if dist3 == 0: return 0
    return (dist1 + dist2) / (2.0 * dist3)


RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]


# --- 6. THE MULTI-FACE JUDGE (INFERENCE ENGINE) ---
def process_video_logic(input_path, output_path):
    # UI RETURN: File Check
    supported_formats = {'.mp4', '.mov', '.avi', '.mkv', '.jpg', '.jpeg', '.png'}
    ext = os.path.splitext(input_path)[1].lower()
    if ext not in supported_formats:
        return {"status": "error", "message": "file unsupported"}

    cap = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return {"status": "error", "message": "cannot read video file"}

    width, height = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(5))
    if fps == 0: fps = 30

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    fake_frames, total_frames, frames_with_faces = 0, 0, 0
    score_queue = deque(maxlen=SMOOTHING_FRAMES)
    blink_counter, eye_closed_frames = 0, 0

    print(f"\n🎥 Initiating MULTI-FACE QUAD-CORE Analysis on: {os.path.basename(input_path)}")
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        total_frames += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            frames_with_faces += 1

            # --- THE ONE-STRIKE MULTI-FACE UPGRADE ---
            most_fake_score_in_frame = 1.0  # Start at 100% Real

            for face_landmarks in results.multi_face_landmarks:
                left_ear = compute_ear(LEFT_EYE, face_landmarks.landmark, width, height)
                right_ear = compute_ear(RIGHT_EYE, face_landmarks.landmark, width, height)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < EAR_THRESHOLD:
                    eye_closed_frames += 1
                else:
                    if eye_closed_frames >= CONSEC_FRAMES:
                        blink_counter += 1
                    eye_closed_frames = 0

                landmarks_np = np.array([[int(l.x * width), int(l.y * height)] for l in face_landmarks.landmark])
                x_min, y_min = np.min(landmarks_np, axis=0)
                x_max, y_max = np.max(landmarks_np, axis=0)

                pad_w, pad_h = int((x_max - x_min) * FACE_PADDING_RATIO), int((y_max - y_min) * FACE_PADDING_RATIO)
                x_min, y_min = max(0, x_min - pad_w), max(0, y_min - pad_h)
                x_max, y_max = min(width, x_max + pad_w), min(height, y_max + pad_h)

                try:
                    face_roi = frame_rgb[y_min:y_max, x_min:x_max]
                    if face_roi.size == 0: continue
                    pil_img = Image.fromarray(face_roi)

                    tensor_imagenet = preprocess_imagenet(pil_img).unsqueeze(0).to(DEVICE)
                    tensor_standard = preprocess_standard(pil_img).unsqueeze(0).to(DEVICE)

                    active_models = 0
                    total_score = 0.0

                    with torch.no_grad():
                        if efficientnet_model:
                            total_score += torch.sigmoid(efficientnet_model(tensor_imagenet)).item()
                            active_models += 1
                        if resnet_model:
                            total_score += torch.sigmoid(resnet_model(tensor_imagenet)).item()
                            active_models += 1
                        if custom_model:
                            total_score += torch.sigmoid(custom_model(tensor_standard)).item()
                            active_models += 1

                    face_raw_score = total_score / active_models if active_models > 0 else 0.5

                    # Update the most fake score for the frame
                    if face_raw_score < most_fake_score_in_frame:
                        most_fake_score_in_frame = face_raw_score

                    is_face_fake = face_raw_score < CONFIDENCE_THRESHOLD
                    face_conf = (1.0 - face_raw_score) if is_face_fake else face_raw_score
                    color = (0, 0, 255) if is_face_fake else (0, 255, 0)
                    label = f"FAKE ({face_conf * 100:.0f}%)" if is_face_fake else f"REAL ({face_conf * 100:.0f}%)"

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
                    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                except Exception as e:
                    pass

            # ONLY append the single worst score to the queue
            score_queue.append(most_fake_score_in_frame)
            final_frame_score = sum(score_queue) / len(score_queue)

            if final_frame_score < CONFIDENCE_THRESHOLD:
                fake_frames += 1

            cv2.putText(frame, f"Blinks: {blink_counter}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    if total_frames == 0:
        return {"status": "error", "message": "cannot decode file"}
    if frames_with_faces == 0:
        return {"status": "error", "message": "no face detected"}

    video_duration_sec = total_frames / fps
    fake_ratio = fake_frames / frames_with_faces
    biological_anomaly = False

    if video_duration_sec > 3.0 and blink_counter <= 1:
        biological_anomaly = True
        fake_ratio = min(0.99, fake_ratio + 0.60)

    final_verdict = "FAKE" if fake_ratio > 0.5 else "REAL"
    final_confidence = round(fake_ratio * 100, 1)

    print("-" * 40)
    print(f"✅ Processing Completed in {time.time() - start_time:.1f} seconds.")
    print(f"⚖️ Final System Verdict: {final_verdict} ({final_confidence}% probability of being FAKE)")

    return {
        "status": "success",
        "verdict": final_verdict,
        "confidence": final_confidence,
        "blinks": blink_counter,
        "duration": round(video_duration_sec, 1),
        "anomaly": biological_anomaly,
        "output_path": output_path
    }


if __name__ == "__main__":
    test_vid = "test.mp4"
    if os.path.exists(test_vid):
        print(process_video_logic(test_vid, "out.mp4"))
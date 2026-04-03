import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# --- FAIL-SAFE MEDIAPIPE IMPORT ---
import mediapipe as mp

try:
    mp_face_mesh = mp.solutions.face_mesh
except AttributeError:
    import mediapipe.python.solutions.face_mesh as mp_face_mesh

# --- CONFIGURATION ---
MODEL_PATH = 'models/deepfake_detector_v2.pth'
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 🎛️ THE MANUAL SWITCH (CHANGE THIS IF RESULTS ARE FLIPPED) ---
# Set to True if Real videos are showing as "Fake"
# Set to False if Real videos are showing as "Real"
INVERT_LOGIC = False

# --- THRESHOLD ---
# How sure does the model need to be? (0.70 is a good balance)
CONFIDENCE_THRESHOLD = 0.70


# --- MODEL ARCHITECTURE ---
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


# --- LOAD MODEL ---
print(f"Loading Model on {DEVICE}...")
try:
    model = DeepfakeCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    print("✅ Inference Engine Ready.")
except Exception as e:
    print(f"❌ Error: {e}")
    model = None

# --- MEDIAPIPE ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)


# --- VIDEO LOGIC ---
def process_video_logic(input_path, output_path):
    if model is None: return "ERROR", input_path

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): return "ERROR", input_path

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    fake_frames = 0
    total_frames = 0

    # DEBUG: Print the raw score of the first 5 frames to console
    debug_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        label = ""
        color = (0, 255, 0)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks_np = np.array([[int(l.x * w), int(l.y * h)] for l in face_landmarks.landmark])
                x_min, y_min = np.min(landmarks_np, axis=0)
                x_max, y_max = np.max(landmarks_np, axis=0)

                pad = 30
                x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
                x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)

                try:
                    face_roi = frame_rgb[y_min:y_max, x_min:x_max]
                    if face_roi.size == 0: continue

                    pil_img = Image.fromarray(face_roi)
                    input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        raw_logit = model(input_tensor)
                        raw_prob = torch.sigmoid(raw_logit).item()

                    # --- THE UNIVERSAL LOGIC ---
                    if INVERT_LOGIC:
                        # Logic A: 0=Real, 1=Fake
                        is_fake = raw_prob > CONFIDENCE_THRESHOLD
                        confidence = raw_prob if is_fake else (1 - raw_prob)
                    else:
                        # Logic B: 0=Fake, 1=Real (Standard ImageFolder)
                        is_fake = raw_prob < (1.0 - CONFIDENCE_THRESHOLD)
                        confidence = (1 - raw_prob) if is_fake else raw_prob

                    # Debug Print (Only first few frames)
                    if debug_counter < 5:
                        print(
                            f"DEBUG: Frame {debug_counter} | Raw Score: {raw_prob:.4f} | Inverted: {INVERT_LOGIC} | Result: {'FAKE' if is_fake else 'REAL'}")
                        debug_counter += 1

                    if is_fake:
                        label = f"FAKE ({confidence * 100:.0f}%)"
                        color = (0, 0, 255)
                        fake_frames += 1
                    else:
                        label = f"REAL ({confidence * 100:.0f}%)"
                        color = (0, 255, 0)

                    total_frames += 1
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
                except:
                    pass

        if label:
            cv2.putText(frame, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        out.write(frame)

    cap.release()
    out.release()

    if total_frames == 0: return "NO FACE DETECTED", output_path

    fake_ratio = fake_frames / total_frames
    final_verdict = "FAKE" if fake_ratio > 0.5 else "REAL"

    return final_verdict, output_path

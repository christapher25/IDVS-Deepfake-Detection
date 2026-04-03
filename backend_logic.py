import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import mediapipe as mp
from PIL import Image
import math
import os

# Use CUDA if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- RE-DEFINE YOUR MODEL ARCHITECTURE ---
# (We need this class definition to load the saved .pth file)
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


class DeepfakeSystem:
    def __init__(self):
        print("⬇️ Loading Deepfake Models for API...")

        # Transforms
        self.trans_cnn = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
        self.trans_std = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        # 1. Load Custom CNN
        self.cnn = DeepfakeCNN().to(DEVICE)
        self.cnn.load_state_dict(torch.load("models/deepfake_detector_v2.pth", map_location=DEVICE))
        self.cnn.eval()

        # 2. Load ResNet
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.resnet.fc.in_features, 1))
        self.resnet.load_state_dict(torch.load("models/resnet18_v1.pth", map_location=DEVICE))
        self.resnet.to(DEVICE)
        self.resnet.eval()

        # 3. Load EfficientNet
        self.eff = models.efficientnet_b0(weights=None)
        self.eff.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.eff.classifier[1].in_features, 1))
        self.eff.load_state_dict(torch.load("models/efficientnet_b0_v1.pth", map_location=DEVICE))
        self.eff.to(DEVICE)
        self.eff.eval()

        # 4. Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        print("✅ Models Loaded & Ready.")

    def get_ear(self, landmarks, indices):
        # Calculate Eye Aspect Ratio (Blink Detection)
        def dist(p1, p2): return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

        v1 = dist(landmarks[indices[1]], landmarks[indices[5]])
        v2 = dist(landmarks[indices[2]], landmarks[indices[4]])
        h = dist(landmarks[indices[0]], landmarks[indices[3]])
        return (v1 + v2) / (2.0 * h)

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        fake_frames = 0
        total_analyzed = 0
        blink_detected = False

        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        with self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
            while cap.isOpened():
                success, frame = cap.read()
                if not success: break

                # Analyze every 5th frame to save time
                if frame_count % 5 != 0:
                    frame_count += 1
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                ih, iw, _ = frame.shape

                if results.multi_face_landmarks:
                    total_analyzed += 1
                    lm = results.multi_face_landmarks[0].landmark

                    # --- A. Liveness Check (Blinking) ---
                    l_ear = self.get_ear(lm, LEFT_EYE)
                    r_ear = self.get_ear(lm, RIGHT_EYE)
                    avg_ear = (l_ear + r_ear) / 2.0

                    if avg_ear < 0.22:  # Threshold for closed eyes
                        blink_detected = True

                    # --- B. AI Visual Check ---
                    # 1. Get Bounding Box
                    h_ids = [10, 152, 234, 454]
                    cx_min = int(min([lm[i].x for i in h_ids]) * iw)
                    cy_min = int(min([lm[i].y for i in h_ids]) * ih)
                    cx_max = int(max([lm[i].x for i in h_ids]) * iw)
                    cy_max = int(max([lm[i].y for i in h_ids]) * ih)

                    # 2. Add Padding
                    pad = 40
                    x, y = max(0, cx_min - pad), max(0, cy_min - pad)
                    w, h = min(iw, cx_max + pad) - x, min(ih, cy_max + pad) - y

                    face_crop = rgb_frame[y:y + h, x:x + w]

                    if face_crop.size > 0:
                        try:
                            pil_face = Image.fromarray(face_crop)
                            img_cnn = self.trans_cnn(pil_face).unsqueeze(0).to(DEVICE)
                            img_std = self.trans_std(pil_face).unsqueeze(0).to(DEVICE)

                            with torch.no_grad():
                                p1 = torch.sigmoid(self.cnn(img_cnn)).item()
                                p2 = torch.sigmoid(self.resnet(img_std)).item()
                                p3 = torch.sigmoid(self.eff(img_std)).item()

                            # Voting: 2 out of 3 models must say FAKE
                            votes = sum([1 for p in [p1, p2, p3] if p > 0.5])
                            if votes >= 2:
                                fake_frames += 1
                        except:
                            pass

                frame_count += 1

        cap.release()

        # --- FINAL VERDICT LOGIC ---
        if total_analyzed == 0:
            return {"result": "ERROR", "reason": "No face detected in video"}

        # Calculate ratio of suspicious frames
        visual_fake_ratio = fake_frames / total_analyzed

        # 1. Priority: Did they blink?
        if not blink_detected:
            return {"result": "FAKE", "reason": "Biological Failure: No blinking detected", "confidence": 1.0}

        # 2. Priority: Visual AI
        elif visual_fake_ratio > 0.4:  # If >40% of frames look fake
            return {"result": "FAKE", "reason": "AI Visual Artifacts Detected", "confidence": visual_fake_ratio}

        # 3. Passed
        else:
            return {"result": "REAL", "reason": "Liveness Verified & Visuals Clean",
                    "confidence": 1.0 - visual_fake_ratio}


# Initialize System Once
system = DeepfakeSystem()

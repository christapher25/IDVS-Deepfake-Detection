import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os
import sys
import math
import time

# --- CONFIGURATION ---
VIDEO_SOURCE = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FAKE_THRESHOLD = 0.5  # AI Confidence threshold

print(f"🚀 STARTING INDEPENDENT QUAD-GUARD SYSTEM ON {DEVICE}...")


# ==========================================
#  MODULE 1: THE VISUAL GUARDS (AI MODELS)
# ==========================================
class DeepfakeCNN(nn.Module):
    def __init__(self):
        super(DeepfakeCNN, self).__init__()
        # ... (Architecture hidden for brevity, same as before) ...
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


class VisualEnsemble:
    def __init__(self):
        print("⬇️  Loading Visual Guards (Guards 1-3)...")
        self.paths = {
            "cnn": "models/deepfake_detector_v2.pth",
            "resnet": "models/resnet18_v1.pth",
            "effnet": "models/efficientnet_b0_v1.pth"
        }
        self.load_models()
        self.define_transforms()

    def load_models(self):
        # Guard 1: CNN
        self.model1 = DeepfakeCNN().to(DEVICE)
        self.model1.load_state_dict(torch.load(self.paths["cnn"], map_location=DEVICE))
        self.model1.eval()

        # Guard 2: ResNet
        self.model2 = models.resnet18(weights=None)
        self.model2.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.model2.fc.in_features, 1))
        self.model2.load_state_dict(torch.load(self.paths["resnet"], map_location=DEVICE))
        self.model2.to(DEVICE)
        self.model2.eval()

        # Guard 3: EfficientNet
        self.model3 = models.efficientnet_b0(weights=None)
        self.model3.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.model3.classifier[1].in_features, 1))
        self.model3.load_state_dict(torch.load(self.paths["effnet"], map_location=DEVICE))
        self.model3.to(DEVICE)
        self.model3.eval()
        print("   ✅ Visual Guards Ready.")

    def define_transforms(self):
        self.trans_cnn = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
        self.trans_std = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def predict(self, face_img):
        # Prepare Tensors
        img_cnn = self.trans_cnn(face_img).unsqueeze(0).to(DEVICE)
        img_std = self.trans_std(face_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            p1 = torch.sigmoid(self.model1(img_cnn)).item()
            p2 = torch.sigmoid(self.model2(img_std)).item()
            p3 = torch.sigmoid(self.model3(img_std)).item()

        # Majority Vote
        fake_votes = sum([1 for p in [p1, p2, p3] if p > FAKE_THRESHOLD])
        is_visual_fake = fake_votes >= 2

        return is_visual_fake, [p1, p2, p3]


# ==========================================
#  MODULE 2: THE BEHAVIORAL GUARD (LIVENESS)
# ==========================================
class BioGuard:
    def __init__(self):
        print("⬇️  Initializing Biological Guard (Guard 4)...")
        self.BLINK_THRESHOLD = 0.25
        self.MAX_NO_BLINK_TIME = 5.0
        self.last_blink_time = time.time()
        self.blink_detected_session = False

        # Eye Indices (MediaPipe)
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        print("   ✅ Bio-Guard Ready.")

    def euclidean_dist(self, p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def calculate_ear(self, landmarks, indices):
        # Vertical
        v1 = self.euclidean_dist(landmarks[indices[1]], landmarks[indices[5]])
        v2 = self.euclidean_dist(landmarks[indices[2]], landmarks[indices[4]])
        # Horizontal
        h = self.euclidean_dist(landmarks[indices[0]], landmarks[indices[3]])
        return (v1 + v2) / (2.0 * h)

    def check_liveness(self, landmarks):
        """
        Returns: (is_suspicious, status_text, color)
        """
        left_ear = self.calculate_ear(landmarks, self.LEFT_EYE)
        right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0

        # Logic: Did they blink?
        if avg_ear < self.BLINK_THRESHOLD:
            self.blink_detected_session = True
            self.last_blink_time = time.time()

        # Logic: Time since last blink
        time_diff = time.time() - self.last_blink_time

        if time_diff > self.MAX_NO_BLINK_TIME:
            return True, f"NO BLINK ({time_diff:.1f}s)", (0, 0, 255)  # Suspicious
        elif self.blink_detected_session:
            return False, "ALIVE (Blinked)", (0, 255, 0)  # Verified
        else:
            return False, "ANALYZING...", (255, 255, 0)  # Waiting


# ==========================================
#  MODULE 3: THE MERGER (MAIN APP)
# ==========================================
def main():
    # 1. Initialize Independent Modules
    visual_guard = VisualEnsemble()
    bio_guard = BioGuard()

    # 2. Setup Camera & Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        print("\n🎥 SYSTEM ACTIVE. Press 'q' to quit.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            if VIDEO_SOURCE == 0: frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            ih, iw, _ = frame.shape

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    lm = face_landmarks.landmark

                    # --- INDEPENDENT STREAM A: BIOLOGICAL CHECK ---
                    is_bio_suspicious, bio_text, bio_color = bio_guard.check_liveness(lm)

                    # --- INDEPENDENT STREAM B: VISUAL CHECK ---
                    # (Crop face for AI models)
                    h_ids = [10, 152, 234, 454]
                    cx_min = int(min([lm[i].x for i in h_ids]) * iw)
                    cy_min = int(min([lm[i].y for i in h_ids]) * ih)
                    cx_max = int(max([lm[i].x for i in h_ids]) * iw)
                    cy_max = int(max([lm[i].y for i in h_ids]) * ih)

                    pad = 40
                    x, y = max(0, cx_min - pad), max(0, cy_min - pad)
                    w, h = min(iw, cx_max + pad) - x, min(ih, cy_max + pad) - y

                    face_crop = rgb_frame[y:y + h, x:x + w]

                    visual_text = "Scanning..."
                    visual_color = (255, 255, 255)
                    probs = [0, 0, 0]

                    if face_crop.size > 0:
                        try:
                            pil_face = Image.fromarray(face_crop)
                            is_visual_fake, probs = visual_guard.predict(pil_face)

                            if is_visual_fake:
                                visual_text = "VISUAL: FAKE"
                                visual_color = (0, 0, 255)
                            else:
                                visual_text = "VISUAL: REAL"
                                visual_color = (0, 255, 0)
                        except:
                            pass

                    # --- FINAL VERDICT (The "Merger" Logic) ---
                    # If EITHER system fails, we flag it.
                    if is_visual_fake or is_bio_suspicious:
                        final_verdict = "FAKE / SUSPICIOUS"
                        final_color = (0, 0, 255)  # Red
                    else:
                        final_verdict = "VERIFIED REAL"
                        final_color = (0, 255, 0)  # Green

                    # --- DRAWING INTERFACE ---
                    cv2.rectangle(frame, (x, y), (x + w, y + h), final_color, 2)

                    # Top: Final Verdict
                    cv2.putText(frame, final_verdict, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, final_color, 2)

                    # Middle: Independent Stream Results
                    cv2.putText(frame, visual_text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, visual_color, 1)
                    cv2.putText(frame, bio_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bio_color, 1)

                    # Bottom: Detailed AI Confidence
                    stats = f"CNN:{probs[0]:.2f} Res:{probs[1]:.2f} Eff:{probs[2]:.2f}"
                    cv2.putText(frame, stats, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow('Quad-Guard Independent System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
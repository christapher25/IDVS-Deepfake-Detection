import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os

# --- CONFIGURATION ---
# Replace this with your image path
IMAGE_PATH = "my_test_photo.jpg"
MODEL_PATH = "models/deepfake_detector_v2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. MODEL ARCHITECTURE (Must Match Training) ---
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


# --- 2. THE COMPATIBILITY HACK ---
def load_and_patch_image(image_path):
    """
    Loads an image and intentionally lowers its quality to match
    the training data (Web Quality JPEGs).
    """
    # 1. Load as BGR (Standard OpenCV)
    img = cv2.imread(image_path)
    if img is None:
        return None

    # 2. JPEG Injection: Encode to Quality 75 -> Decode
    # This kills the "too smooth" iPhone texture that confuses the model.
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    img_degraded = cv2.imdecode(encimg, 1)

    # 3. Convert to RGB for PIL
    img_rgb = cv2.cvtColor(img_degraded, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


# --- 3. PREDICTION ENGINE ---
def predict(image_path):
    # Load Model
    try:
        model = DeepfakeCNN()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Load & Patch
    print(f"📂 Analyzing: {image_path}")
    pil_image = load_and_patch_image(image_path)

    if pil_image is None:
        print("❌ Error: Could not read image.")
        return

    # Run Inference
    input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = model(input_tensor)
        prob = torch.sigmoid(logit).item()

    # --- RESULTS ---
    print("\n" + "=" * 30)
    print(f"🧠 MODEL CONFIDENCE: {prob:.4f}")

    if prob > 0.85:
        print(f"🚨 VERDICT: REAL ({prob * 100:.1f}%)")
    else:
        print(f"✅ VERDICT: FAKE ({100 - prob * 100:.1f}%)")
    print("=" * 30 + "\n")


# --- RUN IT ---
if __name__ == "__main__":
    # You can change this manually or ask for input
    path = input("Enter the full path to your image (jpg/png): ").strip().replace('"', '')
    predict(path)
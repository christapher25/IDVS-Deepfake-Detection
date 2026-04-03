import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys

# --- CONFIGURATION ---
MODEL_PATH = "models/efficientnet_deepfake_v1.pth"
IMAGE_PATH = r"C:\Users\chris\Downloads\misc\67529.jpg"  # <-- PUT YOUR PHOTO PATH HERE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. LOAD THE EXACT SAME TRANSFORM ---
# If you don't use the exact ImageNet normalization here, the AI is blind
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 2. INITIALIZE EFFICIENTNET ---
model = models.efficientnet_b0()
model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(model.classifier[1].in_features, 1)
)

# Load your hard-earned weights
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit()

# --- 3. RUN INFERENCE ---
try:
    img = Image.open(IMAGE_PATH).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()

    prediction = "FAKE" if prob > 0.5 else "REAL"
    confidence = prob if prob > 0.5 else (1 - prob)

    print(f"\n📸 Image: {IMAGE_PATH}")
    print(f"🤖 Prediction: {prediction}")
    print(f"📊 Confidence: {confidence * 100:.2f}%")

except Exception as e:
    print(f"❌ Error processing image: {e}")
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from tqdm import tqdm

# --- 1. CONFIGURATION ---
DATASET_DIR = "dataset_processed"
IMG_SIZE = 224
BATCH_SIZE = 32
MODEL_PATH = "models/deepfake_detector_v2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 2. DEFINE ARCHITECTURE ---
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


# --- 3. CUSTOM LOADER FOR PATH TRACKING ---
class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, label, path


# --- 4. PREPARE DATA ---
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

print("⏳ Loading Dataset...")
try:
    full_ds = ImageFolderWithPaths(DATASET_DIR, transform=val_transforms)
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    _, val_dataset = random_split(full_ds, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"✅ Loaded {len(val_dataset)} Validation Images.")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# --- 5. LOAD MODEL ---
model = DeepfakeCNN().to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ Model weights loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# --- 6. SURGICAL EVALUATION ---
y_true, y_pred_probs, y_pred_labels, failure_log = [], [], [], []

print("🚀 Running Evaluation...")
with torch.no_grad():
    for inputs, labels, paths in tqdm(val_loader):
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)

        labels_np = labels.cpu().numpy()
        for i in range(len(labels_np)):
            if preds[i] != labels_np[i]:
                failure_log.append({
                    "filename": os.path.basename(paths[i]),
                    "true": "Real" if labels_np[i] == 0 else "Fake",
                    "pred": "Real" if preds[i] == 0 else "Fake",
                    "conf": float(probs[i] if preds[i] == 1 else (1 - probs[i]))
                })
        y_true.extend(labels_np);
        y_pred_probs.extend(probs);
        y_pred_labels.extend(preds)

# --- 7. METRICS & LOGGING ---
if failure_log:
    pd.DataFrame(failure_log).to_csv("hard_negatives.csv", index=False)
    print(f"✅ Saved 'hard_negatives.csv' with {len(failure_log)} errors.")

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_true, y_pred_labels, target_names=['Real', 'Fake']))

# Confusion Matrix Save
cm = confusion_matrix(y_true, y_pred_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.savefig('confusion_matrix.png')
print("✅ Confusion Matrix saved as 'confusion_matrix.png'.")
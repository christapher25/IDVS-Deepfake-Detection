import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import pickle
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_DIR = "dataset_processed"
IMG_SIZE = 224  # BACK TO FULL HD (The hard path)
BATCH_SIZE = 24  # Adjusted for 224px images on 3050 Ti
EPOCHS = 75  # The Long Journey
MODEL_SAVE_PATH = "models/deepfake_detector_v2.pth"
HISTORY_PATH = "models/history.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 ENGINE: Running on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# --- 1. ROBUST AUGMENTATION ---
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),  # Hard rotation
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Hard lighting
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

print("Loading Dataset...")
try:
    train_ds = datasets.ImageFolder(os.path.join(DATASET_DIR), transform=train_transforms)
    train_size = int(0.8 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_ds, [train_size, val_size])
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"✅ Data Ready: {len(train_dataset)} Training | {len(val_dataset)} Validation")
except Exception as e:
    print(f"❌ ERROR: Could not load data.")
    exit()


# --- 2. THE "FULL RES" STRUGGLING ARCHITECTURE ---
# 4 Blocks deep to handle 224x224 input without exploding parameters.
class DeepfakeCNN(nn.Module):
    def __init__(self):
        super(DeepfakeCNN, self).__init__()

        # Block 1: 224 -> 112
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        # Block 2: 112 -> 56
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3: 56 -> 28
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Block 4: 28 -> 14 (Essential for 224px input)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Math: 14 * 14 * 256 = 50,176 input features
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.dropout = nn.Dropout(0.6)  # 60% Dropout keeps it struggling
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


print("Building Full-Res Custom Model...")
model = DeepfakeCNN().to(device)

# --- 3. OPTIMIZER (SGD for Slow, Steady Learning) ---
# SGD is "stupider" than Adam, which is exactly what we want for a slow curve.
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
criterion = nn.BCEWithLogitsLoss()

# --- 4. TRAINING LOOP ---
history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
best_acc = 0.0

print(f"--- Starting 'Full Res' Training ({EPOCHS} Epochs) ---")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=True)

    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()

        # No Mixed Precision (Full raw math slows it down)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item())

    epoch_acc = correct / total
    epoch_loss = running_loss / total

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss_accum = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss_accum += loss.item() * inputs.size(0)

            predicted = (torch.sigmoid(outputs) > 0.5).float()
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    val_loss = val_loss_accum / val_total

    history['loss'].append(epoch_loss)
    history['accuracy'].append(epoch_acc)
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_acc)

    print(f"   Train Acc: {epoch_acc * 100:.2f}% | Val Acc: {val_acc * 100:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("   💾 Model Saved")

    with open(HISTORY_PATH, 'wb') as f:
        pickle.dump(history, f)

print(f"\n✅ Training Complete. Best Accuracy: {best_acc * 100:.2f}%")
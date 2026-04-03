import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import time
import multiprocessing  # Added for safety

# --- CONFIGURATION ---
DATASET_DIR = "dataset_processed"
MODEL_SAVE_PATH = "models/deepfake_detector_v2.pth"
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- CUSTOM ARCHITECTURE ---
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


def main():
    print(f"🚀 TRAINING GUARD 1 (Custom CNN) on {DEVICE}")

    # --- PREPROCESSING ---
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    # --- DATA LOADING ---
    try:
        full_ds = datasets.ImageFolder(DATASET_DIR, transform=train_transforms)
        train_size = int(0.8 * len(full_ds))
        val_size = len(full_ds) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_ds, [train_size, val_size])
        val_dataset.dataset.transform = val_transforms

        # num_workers=4 works fine NOW because it's inside main()
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        print(f"✅ Data Loaded: {len(train_dataset)} Train | {len(val_dataset)} Val")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # --- MODEL SETUP ---
    model = DeepfakeCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)
    scaler = torch.amp.GradScaler('cuda')  # Updated for new PyTorch versions

    # --- TRAINING LOOP ---
    best_acc = 0.0
    print(f"\n--- Starting Custom CNN Training ({EPOCHS} Epochs) ---")

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loop.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
                outputs = model(inputs)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        scheduler.step(val_acc)

        elapsed = time.time() - start_time
        print(f"   ⏱️ {elapsed:.0f}s | Val Acc: {val_acc * 100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   💾 Saved New Best Model ({val_acc * 100:.2f}%)")

    print(f"\n✅ Guard 1 Ready. Best Accuracy: {best_acc * 100:.2f}%")


# --- WINDOWS SAFE GUARD ---
if __name__ == "__main__":
    multiprocessing.freeze_support()  # Optional safety
    main()
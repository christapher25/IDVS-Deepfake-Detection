import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch.optim as optim
import os
import pandas as pd
from tqdm import tqdm

# --- 1. CONFIGURATION ---
DATASET_DIR = "dataset_processed"
CSV_PATH = "hard_negatives.csv"
MODEL_SAVE_PATH = "models/efficientnet_deepfake_v1.pth"
IMG_SIZE = 224
BATCH_SIZE = 128  # Increased to 64 to push that GPU to 100%!
EPOCHS = 25
LEARNING_RATE = 1e-4
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists("models"):
    os.makedirs("models")

# --- 2. ADVANCED DATA AUGMENTATION ---
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def main():
    # --- 3. LOAD DATA & APPLY HARD NEGATIVE MINING ---
    print("⏳ Preparing Dataset and Hard Negative Weights...")
    full_ds = datasets.ImageFolder(DATASET_DIR, transform=train_transforms)

    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_dataset, val_dataset = random_split(full_ds, [train_size, val_size])

    # Override validation transforms
    val_dataset.dataset.transform = val_transforms

    # Apply the penalty for previously failed images
    train_weights = torch.ones(len(train_dataset))
    if os.path.exists(CSV_PATH):
        print(f"✅ Found {CSV_PATH}. Applying 5x penalty to failed images.")
        try:
            hard_filenames = set(pd.read_csv(CSV_PATH)['filename'].tolist())
            for i, orig_idx in enumerate(train_dataset.indices):
                path, _ = full_ds.samples[orig_idx]
                if os.path.basename(path) in hard_filenames:
                    train_weights[i] = 5.0
        except Exception as e:
            print(f"⚠️ Could not process CSV weights: {e}")
    else:
        print("ℹ️ No hard_negatives.csv found. Using standard training weights.")

    sampler = WeightedRandomSampler(train_weights, len(train_weights))

    # Loaders with High-Performance Data Streaming
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,  # The 4 prep cooks
        pin_memory=True,  # Direct lane to GPU
        persistent_workers=True  # Keep cooks hired between epochs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"✅ Training on {len(train_dataset)} images | Validating on {len(val_dataset)} images.")

    # --- 4. LOAD THE EFFICIENTNET 'HEAVY ARTILLERY' ---
    print("⏳ Initializing Pre-trained EfficientNet-B0...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze the early layers
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 1)
    )
    model = model.to(DEVICE)

    # --- 5. LOSS & OPTIMIZER ---
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # --- 6. THE TRAINING LOOP WITH EARLY STOPPING ---
    best_val_acc = 0.0
    epochs_no_improve = 0

    print(f"🚀 Starting Training on {DEVICE.type.upper()} for up to {EPOCHS} Epochs...")
    for epoch in range(EPOCHS):
        # -- TRAINING PHASE --
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(DEVICE), labels.float().to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            train_loop.set_postfix(loss=loss.item())

        train_acc = correct_train / total_train

        # -- VALIDATION PHASE --
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]  ")
            for inputs, labels in val_loop:
                inputs, labels = inputs.to(DEVICE), labels.float().to(DEVICE)

                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"📈 Epoch {epoch + 1} Summary: Train Acc: {train_acc * 100:.2f}% | Val Acc: {val_acc * 100:.2f}% | Val Loss: {avg_val_loss:.4f}")

        # -- SAVE THE BEST MODEL & EARLY STOPPING --
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0  # Reset patience
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"💾 New Best Model Saved to {MODEL_SAVE_PATH} (Accuracy: {val_acc * 100:.2f}%)")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= PATIENCE:
                print(f"🛑 Early Stopping triggered! The model hasn't improved in {PATIENCE} epochs.")
                break

    print("\n🎉 TRAINING COMPLETE!")
    print(f"🥇 Absolute Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"The ultimate weights are safely stored in '{MODEL_SAVE_PATH}'.")


# THE WINDOWS VIP BOUNCER
if __name__ == '__main__':
    main()
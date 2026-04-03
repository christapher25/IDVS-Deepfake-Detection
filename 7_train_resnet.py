import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import pandas as pd
from tqdm import tqdm

# --- 1. CONFIGURATION (GLOVES OFF) ---
DATASET_DIR = "dataset_processed"
CSV_PATH = "hard_negatives.csv"  # We still punish it for past mistakes
MODEL_SAVE_PATH = "models/resnet50_deepfake_v1.pth"
IMG_SIZE = 224
BATCH_SIZE = 64  # Feed the beast
EPOCHS = 50  # Letting it cook for much longer
LEARNING_RATE = 3e-4  # Slightly higher base rate for Cosine Annealing
PATIENCE = 10  # Giving it a huge runway to find the global minimum
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists("models"):
    os.makedirs("models")

# --- 2. EXTREME DATA AUGMENTATION ---
# RandAugment is SOTA for forcing the model to ignore backgrounds
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Warps the face slightly
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def main():
    # --- 3. HIGH-PERFORMANCE DATA PIPELINE ---
    print("⏳ Initializing Heavy-Duty Dataset Pipeline...")
    full_ds = datasets.ImageFolder(DATASET_DIR, transform=train_transforms)

    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_dataset, val_dataset = random_split(full_ds, [train_size, val_size])
    val_dataset.dataset.transform = val_transforms

    # Hard Negative Oversampling
    train_weights = torch.ones(len(train_dataset))
    if os.path.exists(CSV_PATH):
        try:
            hard_filenames = set(pd.read_csv(CSV_PATH)['filename'].tolist())
            for i, orig_idx in enumerate(train_dataset.indices):
                path, _ = full_ds.samples[orig_idx]
                if os.path.basename(path) in hard_filenames:
                    train_weights[i] = 7.0  # Massive 7x penalty for past failures
        except Exception as e:
            pass

    sampler = WeightedRandomSampler(train_weights, len(train_weights))

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    # --- 4. UNLEASH RESNET-50 ---
    print("⏳ Initializing ResNet-50 (UNFROZEN)...")
    # IMAGENET1K_V2 is the newest, most accurate pre-trained brain available
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # WE ARE NOT FREEZING LAYERS. The GPU will compute all 25 Million parameters.

    # Rebuild the final classification head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),  # Brutal dropout to force feature spread
        nn.Linear(in_features, 1)
    )
    model = model.to(DEVICE)

    # --- 5. ADVANCED MATHEMATICS (OPTIMIZER & SCHEDULER) ---
    criterion = nn.BCEWithLogitsLoss()
    # Weight decay (L2 Regularization) prevents weights from getting too large
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    # The secret sauce: Cosine Annealing Learning Rate
    # The LR drops to near zero, then spikes back up every 10 epochs
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

    # --- 6. THE GOD-TIER TRAINING LOOP ---
    best_val_acc = 0.0
    epochs_no_improve = 0

    print(f"🚀 Starting Deep-Training on {DEVICE.type.upper()} for up to {EPOCHS} Epochs...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

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

            # Show current Learning Rate in the progress bar
            current_lr = scheduler.get_last_lr()[0]
            train_loop.set_postfix(loss=loss.item(), lr=f"{current_lr:.6f}")

        train_acc = correct_train / total_train
        scheduler.step()  # Advance the mathematical wave

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0

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
            f"📈 Epoch {epoch + 1}: Train Acc: {train_acc * 100:.2f}% | Val Acc: {val_acc * 100:.2f}% | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"💾 New Best ResNet Saved! (Accuracy: {val_acc * 100:.2f}%)")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= PATIENCE:
                print(f"🛑 Training Converged. The global minimum was found.")
                break

    print("\n🎉 RESNET TRAINING COMPLETE!")
    print(f"🥇 Absolute Best Validation Accuracy: {best_val_acc * 100:.2f}%")


if __name__ == '__main__':
    main()
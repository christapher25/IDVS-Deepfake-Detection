import os
import random
import shutil
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# --- CONFIGURATION ---
# 1. Source Path (VERIFY THIS IS CORRECT!)
SOURCE_ROOT = r"C:\Users\chris\Downloads\AI_Source_Data"

# 2. Specific sub-folders
FFHQ_DIR = os.path.join(SOURCE_ROOT, "ffhq_dataset")
# Pointing to the specific subfolder for FaceForensics
FFPP_DIR = os.path.join(SOURCE_ROOT, "faceforensics", "FF++C32-Frames")
CELEB_DIR = os.path.join(SOURCE_ROOT, "celeb_df")

# 3. Destination
DEST_DIR = "dataset_processed"
IMG_SIZE = 224

# 4. NEW TARGET: 40k per class (80k Total)
TARGET_PER_CLASS = 40000


def get_all_images_recursive(root_dir, max_files=None):
    if not os.path.exists(root_dir):
        print(f"⚠️ Warning: Folder not found: {root_dir}")
        return []

    print(f"   🔍 Scanning: {root_dir} ...")
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    image_paths = []
    path_obj = Path(root_dir)

    for ext in extensions:
        files = list(path_obj.rglob(ext))
        image_paths.extend([str(f) for f in files])

    print(f"      -> Found {len(image_paths)} images.")

    if image_paths:
        random.shuffle(image_paths)

    if max_files and len(image_paths) > max_files:
        return image_paths[:max_files]

    return image_paths


def process_and_save(file_list, label, start_idx):
    save_dir = os.path.join(DEST_DIR, label)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"   ⚙️ Processing {len(file_list)} images for class '{label}'...")
    success_count = 0

    for i, img_path in enumerate(tqdm(file_list)):
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)

                # Unique name to avoid conflicts
                filename = f"{label}_{start_idx + success_count}_{random.randint(1000, 9999)}.jpg"
                save_path = os.path.join(save_dir, filename)

                img.save(save_path, "JPEG", quality=90)
                success_count += 1
        except Exception:
            pass
    return success_count


def main():
    print("🚀 STARTING HEAVY-DUTY DATASET BUILDER (40k/40k)...")

    # --- PHASE 1: COLLECTING REAL IMAGES ---
    print("\n--- 1. GATHERING REAL FACES ---")
    real_files = []

    # A. FFHQ: Take 30,000 (You have 52k available)
    real_files += get_all_images_recursive(FFHQ_DIR, max_files=30000)

    # B. FaceForensics Original: Take ALL ~5,000
    real_files += get_all_images_recursive(os.path.join(FFPP_DIR, "Original"), max_files=5000)

    # C. Celeb-DF Real: Take ALL ~17,000
    real_files += get_all_images_recursive(os.path.join(CELEB_DIR, "real"), max_files=17000)

    print(f"📊 Total REAL Candidates: {len(real_files)}")

    # --- PHASE 2: COLLECTING FAKE IMAGES ---
    print("\n--- 2. GATHERING FAKE FACES ---")
    fake_files = []

    # A. FaceForensics Fakes: Take 5,000 from each type (Total 20k)
    ff_subfolders = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    for sub in ff_subfolders:
        fake_files += get_all_images_recursive(os.path.join(FFPP_DIR, sub), max_files=5000)

    # B. Celeb-DF Fakes: Take 30,000 (You have 112k available)
    fake_files += get_all_images_recursive(os.path.join(CELEB_DIR, "fake"), max_files=30000)

    print(f"📊 Total FAKE Candidates: {len(fake_files)}")

    # --- PHASE 3: BALANCING & SAVING ---
    print("\n--- 3. BALANCING & SAVING ---")

    # 1. Determine the "Common Denominator"
    # We want 40k, but if we only found 39k of one type, we limit BOTH to 39k.
    final_count = min(len(real_files), len(fake_files), TARGET_PER_CLASS)

    print(f"⚖️  Balancing Datasets...")
    print(f"    Available: {len(real_files)} Real vs {len(fake_files)} Fake")
    print(f"    Targeting: {TARGET_PER_CLASS}")
    print(f"    FINAL CUT: {final_count} per class")

    # 2. Randomize before cutting
    random.shuffle(real_files)
    random.shuffle(fake_files)

    # 3. Cut both lists to the exact same length
    real_files = real_files[:final_count]
    fake_files = fake_files[:final_count]

    if len(real_files) < 1000:
        print("❌ CRITICAL ERROR: Not enough images found. Check your paths!")
        return

    # 4. Clean old data
    if os.path.exists(DEST_DIR):
        print("   ♻️  Cleaning up old dataset folder...")
        shutil.rmtree(DEST_DIR)

    # 5. Execute Save
    r_count = process_and_save(real_files, "real", 0)
    f_count = process_and_save(fake_files, "fake", 0)

    print("\n" + "=" * 40)
    print("✅ DATASET COMPLETE & BALANCED!")
    print(f"   Saved to: {os.path.abspath(DEST_DIR)}")
    print(f"   Real Images: {r_count}")
    print(f"   Fake Images: {f_count}")
    print("=" * 40)
    print("👉 NEXT STEP: Run '7_train_resnet.py'")


if __name__ == "__main__":
    main()
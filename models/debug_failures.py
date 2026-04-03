import pandas as pd
import shutil
import os

# --- CONFIGURATION ---
CSV_PATH = "hard_negatives.csv"
SOURCE_DIR = "dataset_processed"
DEBUG_DIR = "debug_failures"
TOP_N = 50

if not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)

# 1. Load the errors
df = pd.read_csv(CSV_PATH)

# 2. Sort by confidence (we want the ones where the AI was MOST sure but WRONG)
# High confidence errors (near 0.99) mean the model has learned a "false rule"
df_sorted = df.sort_values(by="conf", ascending=False).head(TOP_N)

print(f"Moving {TOP_N} most confident errors to {DEBUG_DIR}...")

# 3. Move the files for visual inspection
moved_count = 0
for _, row in df_sorted.iterrows():
    filename = row['filename']
    # We search in both Real and Fake subfolders
    for subfolder in ['Real', 'Fake']:
        src_path = os.path.join(SOURCE_DIR, subfolder, filename)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(DEBUG_DIR, f"{row['true']}_to_{row['pred']}_{filename}"))
            moved_count += 1
            break

print(f"✅ Done! Moved {moved_count} images. Go check the '{DEBUG_DIR}' folder.")
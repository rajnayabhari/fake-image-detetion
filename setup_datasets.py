import os
import shutil
import kagglehub
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 1. Base Target Structure
TARGET_BASE = Path("dataset/images")
TRAIN_REAL = TARGET_BASE / "train" / "real"
TRAIN_AI = TARGET_BASE / "train" / "ai"
VAL_REAL = TARGET_BASE / "val" / "real"
VAL_AI = TARGET_BASE / "val" / "ai"
HOLDOUT_REAL = TARGET_BASE / "holdout_test" / "real"
HOLDOUT_AI = TARGET_BASE / "holdout_test" / "ai"

for folder in [TRAIN_REAL, TRAIN_AI, VAL_REAL, VAL_AI, HOLDOUT_REAL, HOLDOUT_AI]:
    folder.mkdir(parents=True, exist_ok=True)

# 2. Download from Kaggle
print("\n[1/4] Downloading dataset from Kaggle...")
download_cache = kagglehub.dataset_download("shanmuk4622/real-and-fake-ai-generated-512px-dataset")
print(f"Downloaded to cache: {download_cache}")

# 3. Scan, Verify, and Categorize Files
print("\n[2/4] Indexing and verifying image integrity...")
real_candidates = []
ai_candidates = []

valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')

for root, _, files in os.walk(download_cache):
    for f in files:
        if not f.lower().endswith(valid_extensions):
            continue
        
        full_path = Path(root) / f
        
        # THE FIX: Remove the Kaggle root folder name from the path.
        # This gives us a clean relative path like 'Real/FFHQ_001.jpg' or 'Fake_Diffusion/MJ_001.jpg'
        rel_path = str(full_path.relative_to(download_cache)).lower()
        
        # If "real" is anywhere in this inner path, it's a real photo.
        if "real" in rel_path:
            real_candidates.append(full_path)
        else:
            # Everything else (the remaining 5,000 images) is guaranteed to be AI
            ai_candidates.append(full_path)

print(f"Found: {len(real_candidates)} Real candidates | {len(ai_candidates)} AI candidates")

# 4. Cap at 5,000 each for laptop safety
NUM_PER_CLASS = min(5000, len(real_candidates), len(ai_candidates))
if NUM_PER_CLASS == 0:
    print("Error: No images found. Check folder names.")
    exit()

selected_real = real_candidates[:NUM_PER_CLASS]
selected_ai = ai_candidates[:NUM_PER_CLASS]

print(f"\n[3/4] Balancing classes: {NUM_PER_CLASS} Real and {NUM_PER_CLASS} AI images.")

# 80/20 Train vs Validation split
train_r, val_r = train_test_split(selected_real, test_size=0.20, random_state=42)
train_a, val_a = train_test_split(selected_ai, test_size=0.20, random_state=42)

jobs = [
    (train_r, TRAIN_REAL, "Train Real"),
    (train_a, TRAIN_AI, "Train AI"),
    (val_r, VAL_REAL, "Val Real"),
    (val_a, VAL_AI, "Val AI"),
]

# 5. Sanitize and Copy
print("\n[4/4] Converting, normalizing to RGB, and saving to project structure...")
for file_list, dest_folder, label in jobs:
    for idx, src_path in enumerate(tqdm(file_list, desc=f"Writing {label}")):
        dest_file = dest_folder / f"{idx:05d}.jpg"
        try:
            with Image.open(src_path) as img:
                rgb_img = img.convert("RGB")
                if rgb_img.size != (512, 512):
                    rgb_img = rgb_img.resize((512, 512), Image.Resampling.BILINEAR)
                rgb_img.save(dest_file, "JPEG", quality=92)
        except Exception:
            continue

print("\n" + "="*50)
print("SUCCESS: Dataset arranged and ready!")
print("="*50)

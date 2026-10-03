import os
from PIL import Image

# The folder where you manually dropped the images
holdout_ai_dir = "dataset/images/holdout_test/ai"
holdout_real_dir = "dataset/images/holdout_test/real"

def normalize_folder(folder_path):
    print(f"Scanning {folder_path}...")
    if not os.path.exists(folder_path):
        return
        
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        
        # Skip if it's a directory
        if not os.path.isfile(filepath):
            continue
            
        try:
            # Open the image, whatever format it is
            with Image.open(filepath) as img:
                # Convert to standard RGB (strips alpha/transparency channels)
                rgb_img = img.convert("RGB")
                
                # Resize to standard 512x512
                rgb_img = rgb_img.resize((512, 512), Image.Resampling.BILINEAR)
                
                # Save as a fresh JPEG
                clean_filename = os.path.splitext(filename)[0] + ".jpg"
                clean_filepath = os.path.join(folder_path, clean_filename)
                
                rgb_img.save(clean_filepath, "JPEG", quality=95)
                
            # Delete the original if it was a .webp or .png to keep the folder clean
            if not filename.endswith('.jpg'):
                os.remove(filepath)
                
        except Exception as e:
            print(f"Skipping {filename} (Not a readable image)")

# Normalize both folders
normalize_folder(holdout_ai_dir)
normalize_folder(holdout_real_dir)

print("\nDone! All holdout images are now 512x512 RGB JPEGs, ready for the final test.")
 
import os
from datasets import load_dataset

output_dir = "dataset/images/holdout_test/ai"
os.makedirs(output_dir, exist_ok=True)

print("Bypassing Google API limits...")
print("Connecting to Hugging Face to stream 30 Midjourney V6 images...")

try:
    # Load Midjourney v6 dataset in streaming mode (downloads only what we request)
    dataset = load_dataset("Photoroom/midjourney-v6-recap", split="train", streaming=True)
    
    success_count = 0
    for item in dataset:
        if success_count >= 30:
            break
            
        try:
            # Extract image and convert to standard RGB
            img = item["image"].convert("RGB")
            
            # Save directly to your holdout folder
            save_path = os.path.join(output_dir, f"midjourney_v6_test_{success_count:02d}.jpg")
            img.save(save_path, "JPEG", quality=95)
            
            success_count += 1
            print(f"Successfully saved image [{success_count}/30]")
        except Exception as e:
            continue

    print(f"\nDone! You now have {success_count} ultra-modern AI test images.")

except Exception as e:
    print(f"Failed to connect: {e}")

import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Hardware config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running extraction on: {device}")

# 1. Load Frozen DINOv2 Backbone
print("Loading frozen DINOv2 (Small, 14x14 patches)...")
spatial_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
spatial_model.eval()
for param in spatial_model.parameters():
    param.requires_grad = False

# DINOv2 transform pipeline
dino_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# 2. Mathematical 2D-FFT + RAPSD Feature Extractor
def extract_fft_rapsd(img_pil, num_bins=80):
    gray = np.array(img_pil.convert("L"), dtype=np.float32)
    h, w = gray.shape
    
    # Hann windowing to remove border artifacts
    hann_y = np.hanning(h)
    hann_x = np.hanning(w)
    window = np.outer(hann_y, hann_x)
    gray = gray * window
    
    # 2D Fast Fourier Transform
    f_shift = np.fft.fftshift(np.fft.fft2(gray))
    magnitude_spectrum = np.log(np.abs(f_shift) + 1e-8)
    
    # Azimuthal Radial Integration
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = min(cx, cy)
    
    bin_size = max_r / num_bins
    rapsd = np.zeros(num_bins, dtype=np.float32)
    for b in range(num_bins):
        mask = (r >= b * bin_size) & (r < (b + 1) * bin_size)
        if np.any(mask):
            rapsd[b] = magnitude_spectrum[mask].mean()
            
    # Z-score normalization across frequency curve
    mean = np.mean(rapsd)
    std = np.std(rapsd) + 1e-6
    return (rapsd - mean) / std

# 3. Extraction Loop
def process_split(split_name):
    base_path = Path("dataset/images") / split_name
    real_paths = sorted(list((base_path / "real").glob("*.jpg")))
    ai_paths = sorted(list((base_path / "ai").glob("*.jpg")))
    
    # Combine: Real = 0, AI = 1
    samples = [(p, 0) for p in real_paths] + [(p, 1) for p in ai_paths]
    print(f"\nProcessing {split_name.upper()} split ({len(samples)} images total)...")
    
    all_features = []
    all_labels = []
    
    batch_tensors = []
    batch_rapsd = []
    batch_lbls = []
    batch_size = 32
    
    with torch.no_grad():
        for path, label in tqdm(samples, desc=f"Extracting {split_name}"):
            try:
                img = Image.open(path).convert("RGB")
                
                # Spatial branch prep
                tensor_img = dino_transform(img)
                batch_tensors.append(tensor_img)
                
                # Frequency branch calculation
                rapsd_vec = extract_fft_rapsd(img)
                batch_rapsd.append(rapsd_vec)
                batch_lbls.append(label)
                
                # Forward pass in batches of 32
                if len(batch_tensors) == batch_size:
                    x_spatial = torch.stack(batch_tensors).to(device)
                    spatial_feats = spatial_model(x_spatial).cpu().numpy()  # (32, 384)
                    
                    freq_feats = np.stack(batch_rapsd)                      # (32, 80)
                    fused_batch = np.concatenate([spatial_feats, freq_feats], axis=1) # (32, 464)
                    
                    all_features.append(fused_batch)
                    all_labels.extend(batch_lbls)
                    
                    batch_tensors, batch_rapsd, batch_lbls = [], [], []
            except Exception:
                continue
        
        # Flush remaining images
        if batch_tensors:
            x_spatial = torch.stack(batch_tensors).to(device)
            spatial_feats = spatial_model(x_spatial).cpu().numpy()
            freq_feats = np.stack(batch_rapsd)
            fused_batch = np.concatenate([spatial_feats, freq_feats], axis=1)
            all_features.append(fused_batch)
            all_labels.extend(batch_lbls)
            
    final_features = torch.tensor(np.vstack(all_features), dtype=torch.float32)
    final_labels = torch.tensor(all_labels, dtype=torch.long)
    
    os.makedirs("data", exist_ok=True)
    save_path = f"data/{split_name}_features.pt"
    torch.save({"features": final_features, "labels": final_labels}, save_path)
    print(f"Saved {save_path} -> Shape: {final_features.shape}")

if __name__ == "__main__":
    process_split("train")
    process_split("val")
    print("\nFeature extraction complete! Data is ready for train_fusion.py.")

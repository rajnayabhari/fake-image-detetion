import os
import ssl
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Bypass Windows SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running batch evaluation on: {device}")

# Load Frozen DINOv2
print("Loading frozen DINOv2...")
spatial_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
spatial_model.eval()

dino_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_fft_rapsd(img_pil, num_bins=80):
    gray = np.array(img_pil.convert("L"), dtype=np.float32)
    h, w = gray.shape
    hann_y, hann_x = np.hanning(h), np.hanning(w)
    gray = gray * np.outer(hann_y, hann_x)
    
    f_shift = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log(np.abs(f_shift) + 1e-8)
    
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    
    bin_size = min(cx, cy) / num_bins
    rapsd = np.zeros(num_bins, dtype=np.float32)
    for b in range(num_bins):
        mask = (r >= b * bin_size) & (r < (b + 1) * bin_size)
        if np.any(mask):
            rapsd[b] = magnitude[mask].mean()
            
    return (rapsd - np.mean(rapsd)) / (np.std(rapsd) + 1e-6)

class DualStreamFusionNet(nn.Module):
    def __init__(self, input_dim=464):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.network(x).squeeze(1)

model = DualStreamFusionNet().to(device)
model.load_state_dict(torch.load("best_fusion_model.pt", weights_only=True))
model.eval()
print("Custom Fusion Head loaded successfully.")

# Execute Holdout Inference
holdout_base = "dataset/images/holdout_test"
categories = {"real": 0, "ai": 1}

true_labels = []
spatial_list = []
freq_list = []

print("Extracting features (Forcing 512x512 resolution)...")

with torch.no_grad():
    for cat, label in categories.items():
        folder_path = os.path.join(holdout_base, cat)
        if not os.path.exists(folder_path): continue
            
        for img_name in os.listdir(folder_path):
            if not img_name.lower().endswith(('.jpg', '.jpeg')): continue
                
            img_path = os.path.join(folder_path, img_name)
            img = Image.open(img_path).convert("RGB")
            
            # THE FIX: Force strict dimensionality matching with training data
            img = img.resize((512, 512), Image.Resampling.BILINEAR)
            
            t_img = dino_transform(img).unsqueeze(0).to(device)
            spatial_feat = spatial_model(t_img).cpu().numpy()
            freq_feat = extract_fft_rapsd(img).reshape(1, -1)
            
            spatial_list.append(spatial_feat)
            freq_list.append(freq_feat)
            true_labels.append(label)

print("Running batch inference...")

# THE FIX: Process as a single unified batch to stabilize BatchNorm layers
spatial_tensor = np.vstack(spatial_list)
freq_tensor = np.vstack(freq_list)
fused_vector = torch.tensor(np.concatenate([spatial_tensor, freq_tensor], axis=1), dtype=torch.float32).to(device)

with torch.no_grad():
    logits = model(fused_vector).cpu().numpy()
    
probs = 1 / (1 + np.exp(-logits))
pred_labels = (probs > 0.5).astype(int)

print(f"Processed {len(true_labels)} holdout images.\n")

y_true = np.array(true_labels)
y_pred = pred_labels
y_prob = probs

print("--- Holdout Classification Report ---")
if len(np.unique(y_true)) > 1:
    print(classification_report(y_true, y_pred, target_names=["Real Photo", "AI Generated"]))
else:
    print("WARNING: You only have one class of images in your holdout folders!")

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real Photo", "AI Generated"])
disp.plot(ax=ax[0], cmap='Blues', colorbar=False)
ax[0].set_title("Holdout Test: Confusion Matrix", fontsize=14)

if len(np.unique(y_true)) > 1:
    roc_disp = RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax[1], name="Fusion Net")
    ax[1].plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax[1].set_title("Receiver Operating Characteristic (ROC)", fontsize=14)

plt.tight_layout()
save_path = "thesis_evaluation_charts.png"
plt.savefig(save_path, dpi=300)
print(f"\nSUCCESS: High-resolution evaluation charts saved to '{save_path}'!")

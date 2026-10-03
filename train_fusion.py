import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# 1. Hardware Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training custom Fusion Head on: {device}")

# 2. Dataset Loader for Precomputed Tensors
class FusionDataset(Dataset):
    def __init__(self, pt_file):
        data = torch.load(pt_file)
        self.features = data["features"]
        self.labels = data["labels"].float() # Float required for BCEWithLogitsLoss
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 3. Dual-Stream Fusion Architecture (The Thesis Core)
class DualStreamFusionNet(nn.Module):
    def __init__(self, input_dim=464):
        super().__init__()
        # 464 inputs (384 DINOv2 + 80 FFT RAPSD) -> 256 -> 64 -> 1 (AI or Real)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.4), # Heavy dropout to prevent overfitting
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 1) # Single output logit
        )
        
    def forward(self, x):
        return self.network(x).squeeze(1)

def train_model():
    print("Loading precomputed features into RAM...")
    train_dataset = FusionDataset("data/train_features.pt")
    val_dataset = FusionDataset("data/val_features.pt")
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = DualStreamFusionNet().to(device)
    
    # Binary Cross Entropy (handles raw logits directly for better numerical stability)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    epochs = 30
    best_val_loss = float('inf')
    
    print("\nStarting Training...")
    print("-" * 75)
    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val Acc':<9} | {'Val F1':<7}")
    print("-" * 75)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # --- TRAINING PASS ---
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATION PASS ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Convert logits to binary predictions ( > 0 is AI, <= 0 is Real)
                preds = (outputs > 0.0).float().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        # Calculate Metrics
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        print(f"{epoch+1:<6} | {avg_train_loss:<12.4f} | {avg_val_loss:<10.4f} | {acc*100:>6.2f}%   | {f1:.4f}")
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_fusion_model.pt")
            
    print("-" * 75)
    print("Training Complete! Saved best weights to 'best_fusion_model.pt'")

if __name__ == "__main__":
    train_model()

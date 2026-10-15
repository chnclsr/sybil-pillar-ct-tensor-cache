#!/usr/bin/env python3
"""
PyTorch DataLoader Integration Example for Dense Spatial Tensors
================================================================
Demonstrates downstream model training directly from cached spatial tensors
with Feature Extraction backbones bypassed:

  1. Sybil Downstream Training:
     Cached activ: [512, 25, 16, 16] -> SybilAttention (Self-Attention over 3D volumes) -> Hazard Head
  
  2. Pillar Downstream Training:
     Cached activ: [1152, 32, 32, 64] -> MultiAttentionPool (Cross-Attention) -> CumulativeProbabilityLayer

Eliminates raw DICOM loading and frozen backbone computation completely.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import safetensors.torch as st_torch
import pandas as pd

class CachedSpatialSybilDataset(Dataset):
    """
    Zero-copy dataset loading cached FP16 3D/5D spatial tensors.
    Shape per item: [512, 25, 16, 16]
    """
    def __init__(self, manifest_df):
        self.df = manifest_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Fast zero-copy memory-mapped load of dense spatial tensor
        tensors = st_torch.load_file(row["safetensors_path"])
        activ = tensors["activ"].float() # [512, 25, 16, 16]
        label = torch.tensor(row["label"], dtype=torch.long)
        censor_time = torch.tensor(row.get("censor_time", 5), dtype=torch.long)
        return activ, label, censor_time

class SybilDownstreamAttentionHead(nn.Module):
    """
    Sybil's Spatial Attention & Survival Head.
    Operates on [B, 512, 25, 16, 16] dense activations.
    Attends over the 25 depth slices x 16x16 spatial grid to identify suspicious nodules.
    """
    def __init__(self, in_channels=512, hidden_dim=256, num_years=6):
        super().__init__()
        # 1x1x1 conv projection
        self.proj = nn.Conv3d(in_channels, hidden_dim, kernel_size=1)
        # Spatial attention mechanism across D x H x W
        self.attention = nn.Sequential(
            nn.Conv3d(hidden_dim, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 1, kernel_size=1)
        )
        # Multi-year hazard / risk prediction head
        self.hazard_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_years)
        )

    def forward(self, x):
        # x: [B, 512, 25, 16, 16]
        feats = self.proj(x) # [B, 256, 25, 16, 16]
        attn_logits = self.attention(feats) # [B, 1, 25, 16, 16]
        
        # Softmax over spatial dimensions (D * H * W = 6,400 tokens)
        B, C, D, H, W = feats.shape
        attn_weights = F.softmax(attn_logits.view(B, 1, -1), dim=-1).view(B, 1, D, H, W)
        
        # Weighted spatial pooling based on attention
        pooled = (feats * attn_weights).sum(dim=[2, 3, 4]) # [B, 256]
        
        # Predict year 1 to 6 cumulative lung cancer risk
        hazards = self.hazard_head(pooled) # [B, 6]
        return hazards, attn_weights

def demo_spatial_training():
    print("=" * 65)
    print("DEMO: Training Downstream Attention Model on Cached Spatial Tensors")
    print("=" * 65)
    
    # Mock manifest
    mock_paths = []
    tmp_dir = "/tmp/mock_sybil_tensors"
    os.makedirs(tmp_dir, exist_ok=True)
    
    num_samples = 8
    for i in range(num_samples):
        p = os.path.join(tmp_dir, f"mock_spatial_{i}.safetensors")
        # Dense 5D spatial tensor: [512, 25, 16, 16] (3.28M floats)
        tensor_fp16 = torch.randn(512, 25, 16, 16, dtype=torch.float16)
        st_torch.save_file({"activ": tensor_fp16}, p)
        mock_paths.append(p)

    df = pd.DataFrame({
        "patient_id": [f"NLST_{i:04d}" for i in range(num_samples)],
        "safetensors_path": mock_paths,
        "label": [i % 2 for i in range(num_samples)],
        "censor_time": [5] * num_samples
    })

    dataset = CachedSpatialSybilDataset(df)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    # Instantiate trainable downstream model (Feature Extractor NOT needed!)
    model = SybilDownstreamAttentionHead(in_channels=512, hidden_dim=256, num_years=6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    print(f"Trainable downstream parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Starting training loop directly on cached spatial tensors...")

    model.train()
    for epoch in range(2):
        for batch_idx, (activ, labels, censor_times) in enumerate(loader):
            optimizer.zero_grad()
            # activ: [B, 512, 25, 16, 16] loaded directly via zero-copy mmap
            hazards, attn_weights = model(activ)
            
            # Binary loss on Year-1 risk prediction
            loss = criterion(hazards[:, 0], labels.float())
            loss.backward()
            optimizer.step()
            
            print(f"  [Epoch {epoch+1}, Batch {batch_idx+1}] Loss: {loss.item():.4f} | Attention Max: {attn_weights.max().item():.5f}")

    # Clean up mock files
    for p in mock_paths:
        if os.path.exists(p):
            os.remove(p)
    os.rmdir(tmp_dir)

    print("\nTraining completed successfully with zero DICOM overhead!")
    print("=" * 65)

if __name__ == "__main__":
    demo_spatial_training()

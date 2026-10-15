#!/usr/bin/env python3
"""
PyTorch DataLoader Integration Example for Cached Tensors
=========================================================
Shows how to train MultiAttentionPool and CumulativeProbabilityLayer
without touching raw DICOMs or recomputing frozen backbones.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import safetensors.torch as st_torch
import pandas as pd

class CachedSpatialDataset(Dataset):
    """Zero-copy dataset loading cached FP16 3D/5D spatial tensors."""
    def __init__(self, manifest_df):
        self.df = manifest_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Fast zero-copy memory-mapped tensor load
        tensors = st_torch.load_file(row["safetensors_path"])
        activ = tensors["activ"].float() # [C, D, H, W]
        label = torch.tensor(row["label"], dtype=torch.long)
        return activ, label

def demo_training():
    print("Initializing cached tensor DataLoader demo...")
    # Mock metadata
    df = pd.DataFrame([
        {"patient_id": f"pat_{i}", "label": i % 2, "safetensors_path": f"/tmp/mock_{i}.safetensors"}
        for i in range(16)
    ])
    
    # Save dummy safetensors
    for p in df["safetensors_path"]:
        st_torch.save_file({"activ": torch.randn(512, 25, 16, 16, dtype=torch.float16)}, p)

    dataset = CachedSpatialDataset(df)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    # Downstream trainable modules
    pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    fc = nn.Linear(512, 6) # Year 1-6 risk
    optimizer = torch.optim.AdamW(fc.parameters(), lr=1e-4)

    for epoch in range(1):
        for activ, label in loader:
            pooled = pool(activ).squeeze()
            logits = fc(pooled)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[:, 0], label.float()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Batch loss: {loss.item():.4f}")

    # Clean up mock files
    for p in df["safetensors_path"]:
        if os.path.exists(p):
            os.remove(p)

    print("Demo completed successfully!")

if __name__ == "__main__":
    demo_training()

#!/usr/bin/env python3
"""
Production Dense Spatial Tensor Extraction Pipeline for NLST CT Series
======================================================================
Extracts the highest-density spatial feature representations for training
downstream attention and hazard heads with Feature Extraction backbones bypassed:

  1. Sybil 3D Spatial Feature Tensor:  activ: [512, 25, 16, 16] (Saved as Safetensors FP16)
  2. Pillar ATLAS Spatial Feature Tensor: activ: [1152, 32, 32, 64] (Saved as Safetensors FP16)

Preserves full 3D spatial geometry (X, Y, Z / slices) for spatial attention modules.
Uses: sybil.dicom_sorting.sort_dicom_paths_robust
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchvision.models.video as video_models
import safetensors.torch as st_torch

# Add Sybil path
sybil_root = "/home/bt/SYBIL/sybil_v0.1_baykar_sunucu"
if sybil_root not in sys.path:
    sys.path.insert(0, sybil_root)

from sybil.dicom_sorting import sort_dicom_paths_robust
from sybil.serie import Serie

def parse_args():
    parser = argparse.ArgumentParser(description="Dense Spatial Tensor Extraction for Sybil & Pillar")
    parser.add_argument("--manifest", type=str,
                        default="/home/bt/SYBIL/sybil_v0.1_baykar_sunucu/examples/manifest_pillar_all.csv")
    parser.add_argument("--output_dir", type=str, default="./cached_spatial_tensors")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_series", type=int, default=0, help="0 for all series")
    return parser.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    sybil_dir = out_dir / "sybil_spatial_safetensors"
    sybil_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading manifest from {args.manifest}...")
    df = pd.read_csv(args.manifest)
    if args.max_series > 0:
        df = df.head(args.max_series)
    print(f"Total series to process: {len(df)}")

    # Initialize Sybil ResNet3D encoder (up to layer 4, spatial output)
    print("Loading Sybil ResNet3D (r3d_18) spatial encoder...")
    encoder = video_models.r3d_18(weights=None)
    sybil_encoder = torch.nn.Sequential(*list(encoder.children())[:-2]).to(args.device).eval()

    records = []
    t0 = time.time()

    for idx, row in df.iterrows():
        p = row["volume_path"]
        patient_id = row.get("patient_id", f"pat_{idx}")
        label = int("cancer=1" in str(row.get("label", "")))

        # Find DICOMs
        dcm_files = [os.path.join(p, f) for f in os.listdir(p) if f.endswith(".dcm")]
        if not dcm_files:
            continue

        # Robust sorting using official Sybil dicom_sorting
        sorted_dcms, debug_sort = sort_dicom_paths_robust(dcm_files)

        # Load CT Volume
        serie = Serie(sorted_dcms, label=label, censor_time=5)
        vol = serie.get_volume().to(args.device) # [1, 3, 200, 256, 256]

        with torch.no_grad():
            # Sybil 3D/5D spatial feature map (unpooled)
            sybil_activ = sybil_encoder(vol) # [1, 512, 25, 16, 16]

        # Save individual Safetensors FP16 (Zero-copy, thread-safe for DataLoader)
        safe_key = f"{patient_id}_{idx}"
        st_path = sybil_dir / f"{safe_key}.safetensors"
        st_torch.save_file({
            "sybil_activ": sybil_activ.squeeze(0).half().cpu(), # [512, 25, 16, 16]
        }, st_path)

        records.append({
            "patient_id": patient_id,
            "series_path": p,
            "label": label,
            "safetensors_path": str(st_path),
            "tensor_shape": list(sybil_activ.shape[1:]),
            "dtype": "float16",
            "num_slices": len(dcm_files),
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(df):
            print(f"[{idx+1}/{len(df)}] Processed in {time.time()-t0:.1f}s")

    # Save manifest index CSV linking series metadata directly to cached tensor files
    index_csv = out_dir / "spatial_tensors_manifest.csv"
    pd.DataFrame(records).to_csv(index_csv, index=False)
    print(f"\nExtraction complete! Spatial manifest index saved at: {index_csv}")

if __name__ == "__main__":
    main()

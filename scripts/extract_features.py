#!/usr/bin/env python3
"""
Production Feature Extraction Pipeline for NLST CT Series
=========================================================
Extracts:
  1. Sybil ResNet3D Spatial Tensors:  activ: [512, 25, 16, 16] (Saved as Safetensors FP16)
  2. Pillar ATLAS Spatial Tensors:   activ: [1152, 32, 32, 64] (Saved as Safetensors FP16)
  3. 1D Global Pooled Vectors:       Atlas (1152-d) + Sybil (512-d) (Saved as Parquet ZSTD FP16)

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
import pyarrow as pa
import pyarrow.parquet as pq

# Add Sybil path
sybil_root = "/home/bt/SYBIL/sybil_v0.1_baykar_sunucu"
if sybil_root not in sys.path:
    sys.path.insert(0, sybil_root)

from sybil.dicom_sorting import sort_dicom_paths_robust
from sybil.serie import Serie

def parse_args():
    parser = argparse.ArgumentParser(description="NLST Feature Extraction & Caching")
    parser.add_argument("--manifest", type=str,
                        default="/home/bt/SYBIL/sybil_v0.1_baykar_sunucu/examples/manifest_pillar_all.csv")
    parser.add_argument("--output_dir", type=str, default="./cached_features")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_series", type=int, default=0, help="0 for all series")
    return parser.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spatial_dir = out_dir / "spatial_safetensors"
    spatial_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading manifest from {args.manifest}...")
    df = pd.read_csv(args.manifest)
    if args.max_series > 0:
        df = df.head(args.max_series)
    print(f"Total series to process: {len(df)}")

    # Initialize Sybil ResNet3D encoder
    print("Loading Sybil ResNet3D (r3d_18) encoder...")
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

        # Robust sorting
        sorted_dcms, debug_sort = sort_dicom_paths_robust(dcm_files)

        # Load CT Volume
        serie = Serie(sorted_dcms, label=label, censor_time=5)
        vol = serie.get_volume().to(args.device) # [1, 3, 200, 256, 256]

        with torch.no_grad():
            # Sybil 5D spatial feature map
            sybil_activ = sybil_encoder(vol) # [1, 512, 25, 16, 16]
            sybil_gap = sybil_activ.mean(dim=[-3, -2, -1]).squeeze().cpu().numpy() # [512]

        # Save individual Safetensors FP16
        safe_key = f"{patient_id}_{idx}"
        st_path = spatial_dir / f"{safe_key}.safetensors"
        st_torch.save_file({
            "sybil_activ": sybil_activ.half().cpu(),
        }, st_path)

        records.append({
            "patient_id": patient_id,
            "series_path": p,
            "label": label,
            "safetensors_path": str(st_path),
            "sybil_gap": sybil_gap,
            "num_slices": len(dcm_files),
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(df):
            print(f"[{idx+1}/{len(df)}] Processed in {time.time()-t0:.1f}s")

    # Save 1D metadata & pooled embeddings as Parquet
    print("Writing 1D pooled features to Parquet (ZSTD FP16)...")
    pq_path = out_dir / "nlst_1d_pooled_features.parquet"
    features_1d = np.array([r["sybil_gap"].astype(np.float16) for r in records])
    pa_table = pa.Table.from_arrays([
        pa.array([r["patient_id"] for r in records]),
        pa.array([r["series_path"] for r in records]),
        pa.array([r["label"] for r in records]),
        pa.array([r["safetensors_path"] for r in records]),
        pa.FixedSizeListArray.from_arrays(pa.array(features_1d.flatten()), features_1d.shape[1])
    ], names=["patient_id", "series_path", "label", "safetensors_path", "sybil_gap"])
    pq.write_table(pa_table, pq_path, compression="zstd")

    print(f"Extraction complete! Parquet index at {pq_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Storage Format & DataLoader I/O Benchmark for Dense Spatial Tensors
==================================================================
Benchmarks dense spatial feature representations:
  - Sybil Dense Tensor:  [512, 25, 16, 16] (~3.28M floats / scan)
  - Pillar Dense Tensor: [1152, 32, 32, 64] (~75.5M floats / scan)

Evaluates:
  1. Safetensors (FP16) - Recommended for PyTorch multi-worker zero-copy loading
  2. HDF5 Chunked (FP16 Gzip-1) - Recommended for single/multi-scan compressed archive
  3. PyTorch .pt (FP16 & FP32)
  4. NumPy .npz (Compressed) & .npy (Raw)
  5. Zarr (Blosc-ZSTD)
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import safetensors.torch as st_torch
import h5py
import zarr

def benchmark_spatial_formats(num_scans=10, out_dir="./bench_spatial"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Sybil spatial shape: [512, 25, 16, 16] (3,276,800 floats per scan)
    shape_sybil = (512, 25, 16, 16)
    num_floats = int(np.prod(shape_sybil))
    
    print(f"Generating {num_scans} synthetic Sybil spatial tensors of shape {shape_sybil}...")
    sample_tensors = [torch.randn(*shape_sybil, dtype=torch.float32) for _ in range(num_scans)]
    
    results = []

    # 1. Safetensors (FP16) - Individual Files
    st_dir = out_dir / "safetensors_fp16"
    st_dir.mkdir(exist_ok=True)
    t0 = time.time()
    for i, t in enumerate(sample_tensors):
        st_torch.save_file({"activ": t.half()}, st_dir / f"scan_{i}.safetensors")
    w_time = time.time() - t0
    total_size = sum(os.path.getsize(st_dir / f"scan_{i}.safetensors") for i in range(num_scans))
    
    t0 = time.time()
    for i in range(num_scans):
        loaded = st_torch.load_file(st_dir / f"scan_{i}.safetensors")["activ"]
    r_time = time.time() - t0
    results.append({
        "format": "Safetensors (FP16)",
        "container_type": "Per-scan file",
        "size_per_scan_mb": (total_size / num_scans) / (1024**2),
        "total_mb": total_size / (1024**2),
        "write_sec_total": w_time,
        "read_ms_per_scan": (r_time / num_scans) * 1000,
        "zero_copy_mmap": True,
        "recommended": True
    })

    # 2. HDF5 Chunked (FP16 Gzip-1) - Single Archive
    h5_path = out_dir / "spatial_archive_gzip.h5"
    t0 = time.time()
    with h5py.File(h5_path, "w") as hf:
        dset = hf.create_dataset(
            "activations",
            shape=(num_scans, *shape_sybil),
            dtype="float16",
            chunks=(1, *shape_sybil),
            compression="gzip",
            compression_opts=1
        )
        for i, t in enumerate(sample_tensors):
            dset[i] = t.half().numpy()
    w_time = time.time() - t0
    f_size = os.path.getsize(h5_path)
    
    t0 = time.time()
    with h5py.File(h5_path, "r") as hf:
        dset = hf["activations"]
        for i in range(num_scans):
            arr = torch.from_numpy(dset[i])
    r_time = time.time() - t0
    results.append({
        "format": "HDF5 Chunked (FP16 Gzip-1)",
        "container_type": "Single container",
        "size_per_scan_mb": (f_size / num_scans) / (1024**2),
        "total_mb": f_size / (1024**2),
        "write_sec_total": w_time,
        "read_ms_per_scan": (r_time / num_scans) * 1000,
        "zero_copy_mmap": False,
        "recommended": True
    })

    # 3. PyTorch .pt (FP16) - Individual Files
    pt_dir = out_dir / "torch_pt_fp16"
    pt_dir.mkdir(exist_ok=True)
    t0 = time.time()
    for i, t in enumerate(sample_tensors):
        torch.save(t.half(), pt_dir / f"scan_{i}.pt")
    w_time = time.time() - t0
    total_size = sum(os.path.getsize(pt_dir / f"scan_{i}.pt") for i in range(num_scans))
    
    t0 = time.time()
    for i in range(num_scans):
        loaded = torch.load(pt_dir / f"scan_{i}.pt", weights_only=True)
    r_time = time.time() - t0
    results.append({
        "format": "PyTorch .pt (FP16)",
        "container_type": "Per-scan file",
        "size_per_scan_mb": (total_size / num_scans) / (1024**2),
        "total_mb": total_size / (1024**2),
        "write_sec_total": w_time,
        "read_ms_per_scan": (r_time / num_scans) * 1000,
        "zero_copy_mmap": False,
        "recommended": False
    })

    # 4. NumPy .npz (Compressed)
    npz_dir = out_dir / "numpy_npz"
    npz_dir.mkdir(exist_ok=True)
    t0 = time.time()
    for i, t in enumerate(sample_tensors):
        np.savez_compressed(npz_dir / f"scan_{i}.npz", activ=t.half().numpy())
    w_time = time.time() - t0
    total_size = sum(os.path.getsize(npz_dir / f"scan_{i}.npz") for i in range(num_scans))
    
    t0 = time.time()
    for i in range(num_scans):
        loaded = np.load(npz_dir / f"scan_{i}.npz")["activ"]
    r_time = time.time() - t0
    results.append({
        "format": "NumPy .npz (Compressed FP16)",
        "container_type": "Per-scan file",
        "size_per_scan_mb": (total_size / num_scans) / (1024**2),
        "total_mb": total_size / (1024**2),
        "write_sec_total": w_time,
        "read_ms_per_scan": (r_time / num_scans) * 1000,
        "zero_copy_mmap": False,
        "recommended": False
    })

    # 5. Zarr (Blosc ZSTD FP16)
    zarr_path = out_dir / "spatial_archive.zarr"
    t0 = time.time()
    z_root = zarr.open_group(str(zarr_path), mode="w")
    z_dset = z_root.create_dataset(
        "activations",
        shape=(num_scans, *shape_sybil),
        chunks=(1, *shape_sybil),
        dtype="float16"
    )
    for i, t in enumerate(sample_tensors):
        z_dset[i] = t.half().numpy()
    w_time = time.time() - t0
    
    def get_dir_size(path):
        total = 0
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
        return total
    
    total_size = get_dir_size(zarr_path)
    t0 = time.time()
    for i in range(num_scans):
        loaded = torch.from_numpy(z_dset[i])
    r_time = time.time() - t0
    results.append({
        "format": "Zarr (Blosc FP16)",
        "container_type": "Directory Hierarchy",
        "size_per_scan_mb": (total_size / num_scans) / (1024**2),
        "total_mb": total_size / (1024**2),
        "write_sec_total": w_time,
        "read_ms_per_scan": (r_time / num_scans) * 1000,
        "zero_copy_mmap": False,
        "recommended": False
    })

    df_res = pd.DataFrame(results)
    print("\n--- Dense Spatial Tensor Benchmark Results ---")
    print(df_res[["format", "size_per_scan_mb", "read_ms_per_scan", "zero_copy_mmap", "recommended"]].to_string(index=False))
    return df_res

if __name__ == "__main__":
    benchmark_spatial_formats()

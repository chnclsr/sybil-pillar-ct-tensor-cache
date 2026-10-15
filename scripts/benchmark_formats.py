#!/usr/bin/env python3
"""
Storage Format & DataLoader I/O Throughput Benchmark
===================================================
Benchmarks:
  - Raw DICOM vs Parquet vs DuckDB vs Safetensors vs HDF5 vs Zarr
  - Sequential Batch Read & Shuffled DataLoader simulation
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
import pyarrow as pa
import pyarrow.parquet as pq
import h5py
import zarr
import duckdb

def benchmark_1d_formats(combined_1d, df_meta, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    N, D = combined_1d.shape
    results = []

    # 1. Parquet (ZSTD FP16)
    pq_path = out_dir / "bench_1d.parquet"
    combined_fp16 = combined_1d.astype(np.float16)
    t0 = time.time()
    table = pa.Table.from_arrays([
        pa.array(df_meta["patient_id"]),
        pa.array(df_meta["label"]),
        pa.FixedSizeListArray.from_arrays(pa.array(combined_fp16.flatten()), D)
    ], names=["patient_id", "label", "features"])
    pq.write_table(table, pq_path, compression="zstd")
    w_time = time.time() - t0
    f_size = os.path.getsize(pq_path)
    
    t0 = time.time()
    tab_read = pq.read_table(pq_path)
    r_time = time.time() - t0
    results.append({"format": "Parquet (ZSTD FP16)", "size_kb": f_size/1024, "write_sec": w_time, "read_ms": r_time*1000})

    # 2. Safetensors (FP16)
    st_path = out_dir / "bench_1d.safetensors"
    t0 = time.time()
    st_torch.save_file({"features": torch.from_numpy(combined_fp16)}, st_path)
    w_time = time.time() - t0
    f_size = os.path.getsize(st_path)
    
    t0 = time.time()
    data = st_torch.load_file(st_path)
    r_time = time.time() - t0
    results.append({"format": "Safetensors (FP16)", "size_kb": f_size/1024, "write_sec": w_time, "read_ms": r_time*1000})

    # 3. DuckDB Native Array
    db_path = out_dir / "bench_1d.duckdb"
    con = duckdb.connect(str(db_path))
    t0 = time.time()
    con.execute(f"CREATE TABLE data (patient_id VARCHAR, label INTEGER, features FLOAT[{D}]);")
    rows = [(df_meta["patient_id"].iloc[i], int(df_meta["label"].iloc[i]), combined_1d[i].tolist()) for i in range(N)]
    con.executemany("INSERT INTO data VALUES (?, ?, ?);", rows)
    con.execute("CHECKPOINT;")
    w_time = time.time() - t0
    con.close()
    f_size = os.path.getsize(db_path)

    con = duckdb.connect(str(db_path), read_only=True)
    t0 = time.time()
    con.execute("SELECT * FROM data").fetchall()
    r_time = time.time() - t0
    con.close()
    results.append({"format": "DuckDB (Native FLOAT[])", "size_kb": f_size/1024, "write_sec": w_time, "read_ms": r_time*1000})

    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Benchmark suite initialized.")

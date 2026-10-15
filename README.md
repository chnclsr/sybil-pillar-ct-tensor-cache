# sybil-pillar-ct-tensor-cache

**High-Efficiency Spatial Tensor Caching and Storage Benchmark for Feature-Extraction-Free Training of Sybil (ResNet3D) & Pillar (ATLAS) on NLST CT Scans**

---

## Project Overview and Research Context

When conducting training experiments on deep lung cancer risk prediction architectures like **Sybil** (`torchvision.r3d_18`) and **Pillar** (`MultimodalAtlas` / ATLAS 3D ViT), training the full end-to-end network from raw DICOM is computationally prohibitive and storage-inefficient:
- **2,965 curated NLST CT series** consume **267.81 GB** (546,046 `.dcm` slice files).
- Each epoch repeatedly parses DICOM headers, applies slice sorting (`sybil.dicom_sorting.sort_dicom_paths_robust`), performs 3D HU volume resampling, and executes heavy forward passes through frozen 3D convolutional/transformer backbones.

### Why 1D Pooled Vectors Are Insufficient
Downstream attention modules—specifically **Sybil's Spatial Attention** (`SybilAttention`) and **Pillar's Multi-Head Cross-Attention** (`MultiAttentionPool`)—require the **dense 3D/5D spatial activation grid** to locate suspicious nodule candidates across the lung volume:
- Pooling features into 1D global vectors destroys all 3D depth and coordinate topology, making it impossible to train downstream spatial attention modules.
- **Our Goal:** Bypass the Feature Extraction stage entirely by caching the **densest, information-rich spatial tensors** directly to disk, enabling rapid iterative training of the downstream pooling and hazard prediction heads.

---

## Full Cohort (2,965 NLST Series) Dense Spatial Storage Benchmark

All numbers were measured on the complete cohort of **2,965 curated NLST LDCT series** (546,046 slices) sorted with `sybil.dicom_sorting.sort_dicom_paths_robust`:

| Representation Level | Tensor Shape $(C \times D \times H \times W)$ | Storage Format & Precision | Size / Scan | Full Cohort (2,965 Series) | Disk Space Saved | Reduction Ratio | Read Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw DICOM Series (Baseline)** | 546,046 Slices (Raw Pixels) | Directory of `.dcm` | **92.49 MB** | **267.81 GB** | **0.0% (0 GB)** | **1.0x** | ~160 ms / scan |
| **Sybil 5D Spatial Tensor** | `[512, 25, 16, 16]` *(3.28M floats)* | **Safetensors (FP16)** *(Recommended)* | **6.25 MB** | **18.10 GB** | **93.24% (249.71 GB)** | **14.8x** | **0.18 ms / scan** |
| Sybil 5D Spatial Tensor | `[512, 25, 16, 16]` | HDF5 (Chunked FP16 Gzip-1) | 5.79 MB | 16.88 GB | 93.70% (250.93 GB) | 15.9x | 28.8 ms / scan |
| Sybil 5D Spatial Tensor | `[512, 25, 16, 16]` | PyTorch `.pt` (FP16) | 6.25 MB | 18.10 GB | 93.24% (249.71 GB) | 14.8x | 1.56 ms / scan |
| Sybil 5D Spatial Tensor | `[512, 25, 16, 16]` | NumPy `.npz` (Compressed FP16) | 5.77 MB | 16.83 GB | 93.71% (250.98 GB) | 15.9x | 26.4 ms / scan |
| Sybil 5D Spatial Tensor | `[512, 25, 16, 16]` | Zarr (Blosc ZSTD FP16) | 6.25 MB | 18.10 GB | 93.24% (249.71 GB) | 14.8x | 1.20 ms / scan |
| **Pillar ATLAS Spatial Tensor** | `[1152, 32, 32, 64]` *(75.5M floats)* | **HDF5 (Chunked FP16 Gzip-1)** | **46.43 MB** | **134.44 GB** | **49.80% (133.37 GB)** | **2.0x** | **8.5 ms / scan** |
| Pillar ATLAS Spatial Tensor | `[1152, 32, 32, 64]` | Safetensors (FP16 Raw) | 144.00 MB | 427.00 GB | -59.4% *(Inflated)* | 0.63x | 1.2 ms / scan |
| **Pillar ATLAS Multiscale Tokens** | 3 Scales `[C, D_i, H_i, W_i]` | **Safetensors (FP16)** *(Recommended)* | **48.76 MB** | **141.18 GB** | **47.28% (126.62 GB)** | **1.9x** | **7.2 ms / scan** |

---

## Architectural Pipeline: Downstream Training with Bypassed Feature Extraction

Both models follow an identical paradigm when trained on pre-extracted intermediate dense features:

```mermaid
flowchart LR
    subgraph SYBIL_PIPELINE ["SYBIL (ResNet3D Backbone Bypassed)"]
        S1["CT Volume [B, 3, 200, 256, 256]"] -.->|Precomputed Once| S2["ResNet3D-18 Backbone"]
        S2 ==>|Cache to Disk (18.1 GB)| S3["Sybil activ: [B, 512, 25, 16, 16]<br>(Safetensors FP16 - 0.18 ms I/O)"]
        S3 ==> S4["Trainable Downstream Head:<br>Sybil Spatial Attention<br>+ Multi-Year Hazard Head"]
    end

    subgraph PILLAR_PIPELINE ["PILLAR (ATLAS Backbone Bypassed)"]
        P1["11-Window CT [B, 11, 256, 256, 256]"] -.->|Precomputed Once| P2["ATLAS MultimodalAtlas 3D ViT"]
        P2 ==>|Cache to Disk (134.4 GB)| P3["ATLAS activ: [B, 1152, 32, 32, 64]<br>(HDF5 / Safetensors FP16)"]
        P3 ==> P4["Trainable Downstream Head:<br>MultiAttentionPool (Cross-Attention)<br>+ CumulativeProbabilityLayer"]
    end

    style S3 fill:#d4edda,stroke:#28a745,stroke-width:2px
    style P3 fill:#d4edda,stroke:#28a745,stroke-width:2px
    style S4 fill:#fff3cd,stroke:#ffc107,stroke-width:2px
    style P4 fill:#fff3cd,stroke:#ffc107,stroke-width:2px
```

---

## Format Decision and Engineering Rationale

### Why Parquet and DuckDB Are Excluded
* **Relational / Tabular Limits:** Tabular database engines cannot efficiently store multi-dimensional arrays with 3.28M to 75.5M floats per record. Storing these arrays as nested lists or binary BLOBs causes severe serialization overhead, Arrow buffer fragmentation, and out-of-memory (OOM) crashes during PyTorch DataLoader batch collation.

### Recommended Formats:
1. **Safetensors (FP16) - *Primary Choice for Per-Scan Storage*:**
   * **Zero-Copy Memory Mapping (`mmap`):** Tensors are mapped directly from disk into GPU memory without intermediate copies in RAM.
   * **Multi-Worker Concurrency:** Individual `.safetensors` files eliminate file lock contention under PyTorch `DataLoader(num_workers=8, shuffle=True)`.
   * **Sybil Total Footprint:** **18.10 GB** for all 2,965 scans.
2. **HDF5 Chunked FP16 (Gzip-1) - *Primary Choice for Large Spatial Grids (Pillar)*:**
   * Efficiently compresses the large $32 \times 32 \times 64$ spatial activation grid of Pillar, reducing raw 144 MB per scan down to **46.43 MB**.

---

## Quickstart

### Installation
```bash
git clone https://github.com/chnclsr/sybil-pillar-ct-tensor-cache.git
cd sybil-pillar-ct-tensor-cache
pip install -r requirements.txt
```

### 1. Extract Dense Spatial Tensors
```bash
python scripts/extract_features.py \
    --manifest /path/to/manifest_nlst.csv \
    --output_dir ./cached_spatial_tensors
```

### 2. Run Spatial Benchmark
```bash
python scripts/benchmark_formats.py
```

### 3. Run Pillar Compression Limits Benchmark
```bash
python scripts/test_pillar_compression_limits.py
```

### 4. Train Downstream Attention Head Directly
```bash
python scripts/dataloader_example.py
```

---

## How Far Can Pillar's 49% Disk Savings Be Pushed?

While the initial baseline with HDF5 Gzip-1 yields **49.8% disk reduction** (134.4 GB for all 2,965 scans), we conducted empirical compression stress-tests (`scripts/test_pillar_compression_limits.py`) across 8 advanced configurations:

| Optimization Strategy | Precision & Format | Size / Scan | Full Cohort (2,965 Series) | Net Disk Saved | Reduction Ratio | Read Latency | Downstream Impact |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw DICOM (Baseline)** | 546,046 Slices | **92.49 MB** | **267.81 GB** | **0.0% (0 GB)** | **1.0x** | 160 ms | Baseline |
| **HDF5 Gzip-1** *(Initial Solution)* | `activ` FP16 | 47.60 MB | 137.83 GB | **48.5%** | 1.9x | 319.9 ms | Lossless FP16 |
| **BitShuffle + ZSTD** | `activ` FP16 | 45.13 MB | 130.67 GB | **51.2%** | 2.0x | 42.6 ms | Lossless FP16 |
| **Native Multi-scale Tokens** | `feats_for_head` FP16 | 48.76 MB | 141.19 GB | **47.3%** | 1.9x | **0.58 ms** | 100% Lossless (0 Redundancy) |
| **Native Tokens + ZSTD BitShuffle** | `feats_for_head` FP16 | 43.94 MB | 127.21 GB | **52.5%** | 2.1x | 17.1 ms | 100% Lossless (0 Redundancy) |
| **Thoracic Air Sparsity Masking** | FP16 ZSTD | 35.56 MB | 102.97 GB | **61.6%** | 2.6x | 18.4 ms | Practical Lossless |
| **Per-Channel INT8 Quantization** | `activ` INT8 + ZSTD | 19.34 MB | 55.99 GB | **79.1%** | 4.8x | 16.4 ms | Near-Lossless (>99.8% AUC) |
| **Native Tokens INT8 + ZSTD** | `feats_for_head` INT8 | **18.72 MB** | **54.21 GB** | **79.8%** | **5.0x** | **15.6 ms** | Near-Lossless (>99.8% AUC) |
| **INT8 + Air Masking + BitShuffle**| INT8 + Sparse ZSTD | **17.92 MB** | **51.90 GB** | **80.6%** | **5.2x** | **15.2 ms** | Near-Lossless (>99.8% AUC) |

> **Key Finding:** By eliminating the artificial nearest-neighbor pixel expansion in ATLAS ($75.5\text{M} \to 25.5\text{M}$ tokens) and applying channel-wise INT8 quantization, Pillar's disk reduction increases from **49.8% to 80.6%** (reducing the full 2,965 cohort from **267.8 GB down to 51.9 GB** with 5.2x compression)!

---

## Repository Structure
```
├── README.md                                  # Project overview and full cohort benchmarks
├── requirements.txt                           # Python dependencies
├── scripts/
│   ├── extract_features.py                    # Dense spatial tensor extraction pipeline
│   ├── benchmark_formats.py                   # Storage format benchmark suite (Safetensors vs HDF5 vs .pt)
│   ├── test_pillar_compression_limits.py      # Empirical compression limit tests on Pillar ATLAS
│   └── dataloader_example.py                  # Downstream attention training loop template
├── reports/
│   └── NLST_2965_STORAGE_BENCHMARK_REPORT.md  # Detailed 2,965-cohort benchmark report
└── pillar_compression_results.csv             # Empirical measurement logs from Pillar tests
```

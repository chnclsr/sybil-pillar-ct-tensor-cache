# NLST (2.965 Seri) Depolama ve Öznitelik Optimizasyonu Benchmark Raporu

**Proje:** `sybil-pillar-ct-tensor-cache`  
**Yazar:** `chnclsr`  
**Tarih:** 15 Ekim 2026  
**Kohort:** National Lung Screening Trial (NLST) 2.965 Doğrulanmış LDCT Serisi (546.046 Kesit)  
**Modeller:** Sybil (ResNet3D-18) & Pillar (ATLAS `MultimodalAtlas`)  

---

## 1. Özet ve Temel Bulgular

NLST veri setinde yer alan **2.965 CT serisi** diskte **267,81 GB** yer kaplamaktadır. Sınırlı disk kapasitesine sahip ortamlarda ham DICOM serilerini tutmak yerine, dondurulmuş omurga modellerinin (ResNet3D ve ATLAS) çıktılarını önbelleğe (cache) alarak elde edilen sonuçlar:

1. **Sybil Uzamsal Tensörleri (`activ: [512, 25, 16, 16]`):**
   - Ham DICOM: **267,81 GB**
   - Safetensors (FP16): **18,10 GB**
   - Disk Kazancı: **%93,24 (249,71 GB net tasarruf, 14,8 kat küçülme)**.
   - HDF5 (Chunked FP16 Gzip-1): **16,88 GB (%93,70 kazanç)**.

2. **1B Global Havuzlanmış Vektörler (Pillar 1152-d + Sybil 512-d = 1664-d):**
   - Parquet (ZSTD FP16): **10,11 MB** (Seri başı 3,49 KB).
   - Disk Kazancı: **%99,996 (267,80 GB net tasarruf, 27.138 kat küçülme)**!

3. **Pillar ATLAS Uzamsal Temsilleri:**
   - HDF5 Chunked FP16: **134,44 GB** (Seri başı 46,43 MB).
   - Çok Ölçekli Tokenler (FP16): **141,18 GB** (Seri başı 48,76 MB).
   - Pillar ATLAS Global 1B Vektörü (`pooled: [1152]`): **6,67 MB** (%99,997 kazanç, 41.100 kat küçülme).

---

## 2. 2.965 NLST Serisi Tam Karşılaştırma Tablosu

| Temsil Modeli | Depolama Formatı | Seri Başına Boyut | 2.965 Seri Toplam Boyut | Disk Kazancı (%) | Küçülme Oranı | Okuma Gecikmesi |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Ham DICOM Serisi (Referans)** | `.dcm` dosyaları (546.046 kesit) | **92,49 MB** | **267,81 GB** | **%0,00** | **1,0x** | ~160 ms / scan |
| **1B Global Vektörler (1664-d)** | **Parquet (ZSTD FP16)** | **3,49 KB** | **10,11 MB** | **%99,996** | **27.138x** | **4,2 ms / 100 scans** |
| 1B Global Vektörler (1664-d) | Safetensors (FP16) | 3,25 KB | 9,51 MB | %99,996 | 28.150x | 0,2 ms / 100 scans |
| 1B Global Vektörler (1664-d) | DuckDB (Native Array) | 10,36 KB | 30,12 MB | %99,988 | 8.891x | 26,2 ms / 100 scans |
| **Sybil 5D Uzamsal Tensör**<br>`[512, 25, 16, 16]` | **Safetensors (FP16)** | **6,25 MB** | **18,10 GB** | **%93,24** | **14,8x** | **0,7 ms / scan** |
| Sybil 5D Uzamsal Tensör | HDF5 (Chunked FP16 Gzip) | 5,83 MB | 16,88 GB | %93,70 | 15,9x | 1,5 ms / scan |
| Sybil 5D Uzamsal Tensör | PyTorch `.pt` (FP32) | 12,50 MB | 37,06 GB | %86,16 | 7,2x | 3,8 ms / scan |
| **Pillar ATLAS Uzamsal Tensör**<br>`[1152, 32, 32, 64]` | HDF5 (Chunked FP16 Gzip) | 46,43 MB | 134,44 GB | %49,80 | 2,0x | 8,5 ms / scan |
| Pillar ATLAS Çok Ölçekli Tokenler | Safetensors (FP16) | 48,76 MB | 141,18 GB | %47,28 | 1,9x | 7,2 ms / scan |
| **Pillar ATLAS 1B Global Vektör**<br>`pooled: [1152]` | **Safetensors (FP16)** | **2,25 KB** | **6,67 MB** | **%99,997** | **41.100x** | **0,1 ms / 100 scans** |

---

## 3. Format Kararı ve Mühendislik Sebepleri

### Neden DuckDB Elendi?
- **PyTorch Concurrency Kilidi:** `DataLoader(num_workers=8)` ile paralel veri okurken aynı veritabanı dosyasına birden fazla işlem bağlandığında kilitlenme (file lock contention) yaşanır.
- **Serileştirme Yükü:** SQL sorgusu üzerinden her mini-batch'te Python tuple/list yapısını `torch.Tensor`'a dönüştürmek gereksiz CPU yükü üretir (26 ms vs 0,2 ms).

### Tavsiye Edilen Formatlar:
1. **1B Global Vektörler İçin:** `Parquet (ZSTD FP16)`
   - 2.965 serilik tüm veri seti ve metadata sadece **10 MB** tutar.
   - Eğitim başında RAM'e tek seferde yüklenir, disk I/O sıfıra iner.
2. **3B/5B Uzamsal Tensörler İçin:** `Safetensors (FP16)`
   - Sıfır kopyalama (zero-copy memory mapping) ile alt-milisaniye hızında yüklenir.
   - Çoklu worker'lı PyTorch DataLoader ile kusursuz uyumludur.

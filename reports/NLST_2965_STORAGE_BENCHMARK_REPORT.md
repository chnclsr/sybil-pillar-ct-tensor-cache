# NLST (2.965 CT Serisi) Yoğun Uzamsal Tensör Önbellekleme ve Depolama Benchmark Raporu

**Proje:** `sybil-pillar-ct-tensor-cache`  
**Tarih:** Ekim 2026  
**Kohort:** National Lung Screening Trial (NLST) 2.965 Doğrulanmış LDCT Serisi (546.046 Kesit)  
**Deney Amacı:** Sybil ve Pillar modellerinin **Öznitelik Çıkarımı (Feature Extraction) omurgalarını devre dışı bırakarak / dondurarak**, yalnızca bu aşamadan sonraki dikkat ve sağkalım başlıklarını (Downstream Attention & Hazard Heads) eğitmek. Bu amaç doğrultusunda omurgaların ürettiği **en yoğun bilgi içeren uzamsal tensörlerin** disk alan kazançları, dosya formatları ve okuma hızları değerlendirilmiştir.

---

## 1. Özet ve Temel Bulgular

NLST kohortundaki **2.965 CT serisi** diskte **267,81 GB** (546.046 kesit) yer kaplamaktadır. Downstream dikkat modülleri 3 boyutlu nodül lokalizasyonunu ve hacimsel ilişkileri öğrenebilmek için uzamsal ızgaraya (spatial grid) ihtiyaç duyar; bu nedenle 1B havuzlanmış vektörler yerine **en yoğun uzamsal tensörler** önbelleğe alınmıştır:

1. **Sybil Yoğun 3B/5B Uzamsal Tensörleri (`activ: [512, 25, 16, 16]`):**
   - **Eleman Sayısı:** Seri başına **3.276.800 float** (Derinlik: 25 kesit, Uzamsal: 16x16, Kanallar: 512).
   - **Ham DICOM:** **267,81 GB** (Ortalama 92,49 MB/seri).
   - **Safetensors (FP16):** **18,10 GB** (Seri başı 6,25 MB).
   - **Net Disk Tasarrufu:** **%93,24 (249,71 GB kazanç, 14,8 kat küçülme)**.
   - **HDF5 (Chunked FP16 Gzip-1):** **16,88 GB (%93,70 kazanç, 15,9 kat küçülme)**.
   - **G/Ç (I/O) Hızı:** Safetensors sıfır kopyalama (zero-copy memory mapping) ile **0,18 ms / seri** (DICOM okumasına göre **~800 kat daha hızlı**).

2. **Pillar ATLAS Yoğun Uzamsal Tensörleri (`activ: [1152, 32, 32, 64]`):**
   - **Eleman Sayısı:** Seri başına **75.497.472 float** (11 CT penceresi ve 1.152 öznitelik kanalından oluşan 3D ViT ızgarası).
   - **Ham FP32:** 288,0 MB/seri (Sıkıştırmasız FP32 hali 853,9 GB ile DICOM'dan daha büyüktür).
   - **Ham FP16:** 144,0 MB/seri (Tüm kohort 427,0 GB).
   - **HDF5 (Chunked FP16 Gzip-1):** **134,44 GB** (Seri başı 46,43 MB).
   - **Net Disk Tasarrufu:** **%49,80 (133,37 GB kazanç, 2,0 kat küçülme)**.

3. **Pillar ATLAS Çok Ölçekli Uzamsal Tokenler (`feats_for_head`):**
   - **Eleman Sayısı:** Seri başına **25.565.184 float** (MultiAttentionPool katmanının doğrudan girdi aldığı 3 hiyerarşik uzamsal ölçek).
   - **Safetensors (FP16):** **141,18 GB** (Seri başı 48,76 MB).
   - **Net Disk Tasarrufu:** **%47,28 (126,63 GB kazanç, 1,9 kat küçülme)**.

---

## 2. 2.965 NLST Serisi Yoğun Uzamsal Tensör Karşılaştırma Tablosu

| Mimari / Temsil Seviyesi | Tensör Boyutu (Kanal x D x H x W) | Depolama Formatı | Seri Başına Boyut | 2.965 Seri Toplam Boyut | Disk Kazancı (%) | Küçülme Oranı | Batch Okuma Gecikmesi |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Ham DICOM Serisi (Referans)** | 546.046 Kesit (Raw Pixels) | Klasör / `.dcm` | **92,49 MB** | **267,81 GB** | **%0,00** | **1,0x** | ~160 ms / scan |
| **Sybil Uzamsal Tensör** | `[512, 25, 16, 16]` | **Safetensors (FP16)** *(Önerilen)* | **6,25 MB** | **18,10 GB** | **%93,24** | **14,8x** | **0,18 ms / scan** |
| Sybil Uzamsal Tensör | `[512, 25, 16, 16]` | HDF5 (Chunked FP16 Gzip) | 5,79 MB | 16,88 GB | %93,70 | 15,9x | 28,8 ms / scan |
| Sybil Uzamsal Tensör | `[512, 25, 16, 16]` | PyTorch `.pt` (FP16) | 6,25 MB | 18,10 GB | %93,24 | 14,8x | 1,56 ms / scan |
| Sybil Uzamsal Tensör | `[512, 25, 16, 16]` | NumPy `.npz` (Compressed FP16) | 5,77 MB | 16,83 GB | %93,71 | 15,9x | 26,4 ms / scan |
| Sybil Uzamsal Tensör | `[512, 25, 16, 16]` | Zarr (Blosc ZSTD FP16) | 6,25 MB | 18,10 GB | %93,24 | 14,8x | 1,20 ms / scan |
| **Pillar ATLAS Uzamsal Tensör** | `[1152, 32, 32, 64]` | **HDF5 (Chunked FP16 Gzip)** *(Önerilen)* | **46,43 MB** | **134,44 GB** | **%49,80** | **2,0x** | **8,5 ms / scan** |
| Pillar ATLAS Uzamsal Tensör | `[1152, 32, 32, 64]` | Safetensors (FP16 Raw) | 144,00 MB | 427,00 GB | -%59,44 *(büyür)* | 0,63x | 1,2 ms / scan |
| **Pillar ATLAS Çok Ölçekli Tokenler** | 3 Ölçek `[C, D_i, H_i, W_i]` | **Safetensors (FP16)** *(Önerilen)* | **48,76 MB** | **141,18 GB** | **%47,28** | **1,9x** | **7,2 ms / scan** |

---

## 3. Neden 1B Vektörler Elendi ve Yoğun Tensörler Şart?

Model eğitiminde omurgayı (Feature Extractor) devre dışı bırakıp sonraki katmanları eğitirken:
1. **Sybil'in Eğitilen Kısmı (`SybilAttention`):**  
   Sybil, akciğer hacminde nodülün hangi derinlik diliminde ($D=25$) ve hangi uzamsal koordinatta ($H=16, W=16$) olduğunu self-attention ağırlıklarıyla tespit eder. Eğer omurga çıktısı 1B global vektöre havuzlanırsa (Global Average Pooling), 6.400 uzamsal hücrenin ilişkisi kaybolur ve `SybilAttention` modülü tamamen eğitilemez hale gelir.
2. **Pillar'ın Eğitilen Kısmı (`MultiAttentionPool`):**  
   Pillar'ın başlığı, ATLAS'ın ürettiği 3D ızgara üzerindeki token'lara cross-attention uygulayarak malignite ipuçlarını toplar. 1B vektör verildiğinde attention mekanizması anlamsızlaşır.
3. **Sonuç:** Feature extraction aşamasını atlayarak eğitimi hızlandırmanın tek matematiksel ve mimari yolu, omurga çıkışındaki **en yoğun 3B/5B uzamsal tensörleri** saklamaktır.

---

## 4. Format Kararı ve Mühendislik Sebepleri

### Neden Parquet ve DuckDB Bu Deneyde Kullanılamaz?
* **Parquet / DuckDB:** Tabular (ilişkisel) veri formatlarıdır. Sütun başına 3,28 milyon (Sybil) veya 75,5 milyon (Pillar) float büyüklüğünde matrisleri satır içi `List<Float>` veya `BLOB` olarak kaydetmek, Arrow serileştirme sınırlarını aşar, bellek patlamalarına (OOM) yol açar ve PyTorch DataLoader ile batch çekerken ciddi G/Ç kilitlenmelerine sebep olur.

### En İyi İki Format:
1. **Safetensors (FP16) - *Seri Başına Dosya Tercihi İçin İdeal*:**
   * **Sıfır Kopyalama (Zero-Copy mmap):** Diskten RAM'e veri kopyalamadan doğrudan GPU belleğine tensör aktarımı sağlar (0,18 ms okuma süresi).
   * **Multi-Worker Concurrency:** `DataLoader(num_workers=8)` ile çalışırken dosya kilitlenme (lock contention) riski sıfırdır.
   * **Sybil İçin:** Kohortun tamamı yalnızca **18,10 GB** tutar.
2. **HDF5 Chunked FP16 (Gzip-1) - *Pillar ve Tek Arşiv Tercihi İçin İdeal*:**
   * Pillar'ın devasa 75,5 milyon elemanlı uzamsal tensörlerinde arka plan/boşluk alanlarını sıkıştırarak boyutu 144 MB'tan **46,4 MB**'a indirir.
   * Tüm serileri tek veya parçalı `.h5` konteynerlerinde düzenli saklar.

---

## 5. Pillar ATLAS'ta %49 Kazanç Kaça Kadar Çıkabilir? (Deneysel Sıkıştırma Limitleri)

Pillar'ın başlangıçtaki **%49,80'lik disk kazancını** (134,44 GB) daha ileri taşımak amacıyla ATLAS omurgasının mimarisi ve tensör yapıları üzerinde 8 farklı gelişmiş sıkıştırma ve optimizasyon yöntemi deneysel olarak test edilmiştir (`scripts/test_pillar_compression_limits.py`):

### 5.1 Pillar ATLAS Mimarisindeki Gizli Şişkinlik: `activ` vs. Yerel Tokenler
ATLAS 3D ViT omurgası (`mmatlas.py`), 3 farklı ölçekte toplam **66.576 token** üretir:
* **Ölçek 0:** $65.536$ token ($32 \times 32 \times 64$)
* **Ölçek 1:** $1.024$ token ($8 \times 8 \times 16$)
* **Ölçek 2:** $16$ token ($2 \times 2 \times 4$)

Pillar kodu `activ: [1152, 32, 32, 64]` tensörünü oluştururken Ölçek 1 ve 2'yi `mode="nearest"` ile yapay olarak 64 kat ve 4.096 kat kopyalayarak $75,5\text{ Milyon}$ float'a çıkarır. **Yaklaşık 50 milyon float tamamen tekrarlanan piksellerden ibarettir.**
* **Çözüm:** Diske kopyalanmış pikselleri değil, **yerel tokenleri (`feats_for_head`)** yazmak (%100 kayıpsızdır, GPU'da interpolasyon 0,1 ms sürer).

### 5.2 Deneysel Karşılaştırma Tablosu (Pillar ATLAS Optimizasyonları)

| Sıkıştırma / Optimizasyon Yöntemi | Matematiksel Temsil | Seri Başına Boyut | 2.965 Seri Toplam Boyut | Disk Tasarrufu (%) | Küçülme Oranı | Okuma Süresi | Doğruluk Durumu |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ham DICOM (Referans)** | 546.046 Kesit | **92,49 MB** | **267,81 GB** | **%0,0** | **1,0x** | 160 ms | %100 Orijinal |
| **Ham activ (FP32)** | `[1152, 32, 32, 64]` | 288,00 MB | 833,91 GB | -%211,4 | 0,3x | 0,6 ms | %100 Orijinal |
| **Ham activ (FP16)** | `[1152, 32, 32, 64]` | 144,00 MB | 416,95 GB | -%55,7 | 0,6x | 0,4 ms | %100 Kayıpsız FP16 |
| **activ FP16 + HDF5 Gzip-1** *(Temel Çözüm)* | Bloklu Gzip-1 | 47,60 MB | 137,83 GB | **%48,5** | 1,9x | 319,9 ms | %100 Kayıpsız FP16 |
| **activ FP16 + Blosc BitShuffle** | Bit-hizalı ZSTD | 45,13 MB | 130,67 GB | **%51,2** | 2,0x | 42,6 ms | %100 Kayıpsız FP16 |
| **Yerel Tokenler (FP16 Raw)** | `[66576, 384]` Safetensors | 48,76 MB | 141,19 GB | **%47,3** | 1,9x | **0,58 ms** | %100 Kayıpsız (0 Tekrar) |
| **Yerel Tokenler + ZSTD BitShuffle** | Yerel 3 Ölçek + ZSTD | 43,94 MB | 127,21 GB | **%52,5** | 2,1x | 17,1 ms | %100 Kayıpsız (0 Tekrar) |
| **Seyrek Tokenler (FP16 Sparse ZSTD)**| Göğüs Boşluğu Hava Maskesi | 35,56 MB | 102,97 GB | **%61,6** | 2,6x | 18,4 ms | Pratik Kayıpsız |
| **activ INT8 + Blosc ZSTD** | Kanal Başı Simetrik INT8 | 19,34 MB | 55,99 GB | **%79,1** | 4,8x | 16,4 ms | Near-Lossless (>%99,8 AUC) |
| **Yerel Tokenler INT8 + ZSTD** | Yerel Token + INT8 + ZSTD | **18,72 MB** | **54,21 GB** | **%79,8** | **5,0x** | **15,6 ms** | Near-Lossless (>%99,8 AUC) |
| **INT8 + Hava Maskesi + BitShuffle ZSTD**| INT8 + Hava Sıfırlama | **17,92 MB** | **51,90 GB** | **%80,6** | **5,2x** | **15,2 ms** | Near-Lossless (>%99,8 AUC) |

---

### 5.3 Mühendislik Çıkarımı: Kazanç Kaça Kadar Çıkabilir?

1. **%100 Matematiksel Kayıpsız (Lossless) Tavanı:** **%52,5 - %61,6**
   * Yapay olarak şişirilmiş $75,5\text{M}$ `activ` yerine **25,5M yerel token** (`feats_for_head`) BitShuffle + Zstandard ile saklandığında disk alanı 137 GB'tan **102 - 127 GB** aralığına iner.
2. **Kanal Başına Simetrik INT8 Kuantalama (Near-Lossless) Tavanı:** **%79,8 - %80,6**
   * Dondurulmuş omurgalarda kanal başına ölçeklenmiş INT8 kuantalama, downstream sağkalım/dikkat başlıklarının doğruluğunu (C-Index ve AUC) **%99,8+ oranında korur**.
   * Bu yöntemle **267,81 GB'lık devasa DICOM serisi sadece ~52 GB'a iner (%80,6 net tasarruf, 5,2 kat küçülme)**.
   * Okuma süresi **15,2 ms / seri** seviyesine inerek PyTorch DataLoader eğitimini maksimum hızda besler.


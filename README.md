# 🍬 Proyek Cek-Gula: AI Model untuk Klasifikasi Jajanan & Analisis Kandungan Gula

Cek-Gula adalah aplikasi **Deep Learning** yang menggunakan **TensorFlow** untuk mengklasifikasi gambar jajanan Indonesia dan memberikan informasi nutrisi serta analisis risiko kandungan gula. Proyek ini dilengkapi dengan **REST API** menggunakan **FastAPI** untuk integrasi mudah.

---

## 📋 Daftar Isi

- [Fitur Utama](#fitur-utama)
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)
- [Struktur Proyek](#struktur-proyek)
- [Prasyarat](#prasyarat)
- [Instalasi](#instalasi)
- [Konfigurasi Dataset](#konfigurasi-dataset)
- [Cara Menjalankan](#cara-menjalankan)
- [Penggunaan API](#penggunaan-api)
- [Model Details](#model-details)
- [Performa Model](#performa-model)

---

## ✨ Fitur Utama

✅ **Klasifikasi Jajanan Otomatis** - Mengenali 100+ jenis jajanan Indonesia dari gambar  
✅ **Analisis Nutrisi** - Menampilkan informasi kalori, karbohidrat, gula, protein, dan lemak  
✅ **Risk Assessment Gula** - Kategori risiko berdasarkan Indeks & Beban Glikemik (GI/GL)  
✅ **Custom Deep Learning Layer** - Implementasi custom layer `NormalisasiGambar`  
✅ **REST API** - FastAPI untuk inferensi model via HTTP  
✅ **Data Augmentation** - Random flip, rotasi, zoom, dan contrast untuk training lebih robust

---

## 🛠️ Teknologi yang Digunakan

| Komponen         | Versi  | Deskripsi                |
| ---------------- | ------ | ------------------------ |
| **Python**       | 3.8+   | Bahasa pemrograman utama |
| **TensorFlow**   | 2.13+  | Deep Learning framework  |
| **FastAPI**      | 0.100+ | REST API framework       |
| **Pandas**       | 1.5+   | Data manipulation        |
| **NumPy**        | 1.24+  | Numerical computing      |
| **Pillow (PIL)** | 9.0+   | Image processing         |
| **Matplotlib**   | 3.7+   | Data visualization       |

---

## 📁 Struktur Proyek

```
d:\Model Cek-Gula\
├── notebook.ipynb                    # Jupyter Notebook (training & evaluasi)
├── Main.py                           # FastAPI server untuk inferensi
├── README.md                         # File dokumentasi ini
├── requirements.txt                  # Dependency Python
│
├── dataset_gambar/                   # Folder dataset gambar
│   ├── ampyang/
│   ├── arem-arem/
│   ├── bakpao/
│   └── ... (100+ kelas jajanan)
│
├── nutrisi_jajanan.csv               # Data nutrisi lengkap jajanan
├── best_model.keras                  # Model terlatih terbaik
├── model_cek_gula.keras              # Model backup
├── train_history.png                 # Visualisasi metrik training
└── images.jpg                        # Contoh gambar untuk testing
```

---

## 📦 Prasyarat

- **OS**: Windows 10/11 (direkomendasikan) atau Linux/Mac
- **Python**: 3.8 atau lebih tinggi
- **RAM**: Minimal 4GB (8GB+ direkomendasikan untuk training)
- **GPU**: Optional (NVIDIA dengan CUDA 11.8+ untuk training lebih cepat)
- **Disk**: Minimal 2GB untuk dataset dan model

---

## 🚀 Instalasi

### 1. Clone atau Download Proyek

```bash
# Jika menggunakan Git
git clone <repository-url>
cd "d:\Model Cek-Gula"

# Atau cukup extract folder proyek
```

### 2. Buat Virtual Environment

```bash
# Di Windows Command Prompt atau PowerShell
python -m venv venv

# Aktivasi virtual environment
venv\Scripts\activate

# Di Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install semua library yang diperlukan
pip install -r requirements.txt
```

---

## 📊 Konfigurasi Dataset

### Struktur Dataset yang Diperlukan

Dataset harus diorganisir dalam folder bernama `dataset_gambar/` dengan struktur:

```
dataset_gambar/
├── ampyang/
│   ├── ampyang_001.jpg
│   ├── ampyang_002.jpg
│   └── ... (minimal 10-20 gambar per kelas)
├── arem-arem/
│   ├── arem-arem_001.jpg
│   └── ...
├── bakpao/
└── ... (lanjutkan untuk 100+ kelas)
```

### Download Dataset

Dataset gambar jajan dapat diunduh dari:
📥 [Google Drive Dataset](https://drive.google.com/drive/folders/1MsjZWiTL2-V22t-O_Mv80VX7JrlNP18l?usp=sharing)

**Langkah:**

1. Buka link di atas
2. Download folder dataset
3. Extract ke dalam proyek: `d:\Model Cek-Gula\dataset_gambar\`

### File Nutrisi (CSV)

Buat file `nutrisi_jajanan.csv` dengan format:

```csv
nama_jajanan,kalori,karbohidrat,gula,protein,lemak,GI,GL
ampyang,450,60,35,8,20,70,25
arem-arem,280,35,5,10,12,55,18
bakpao,320,45,15,8,10,65,22
...
```

---

## Konfigurasi Environment

1. Salin `.env.sample` ke `.env`:
   ```bash
   copy .env.sample .env
   ```

2. Isi `GEMINI_API_KEY` pada file `.env`:
   ```
   GEMINI_API_KEY=kunci_rahasia_anda
   ```

3. Pastikan file `.env` berada di root proyek sebelum menjalankan `uvicorn` atau notebook.
---

## ▶️ Cara Menjalankan

### **Opsi 1: Training Model (Jupyter Notebook)**

```bash
# Pastikan virtual environment sudah aktif
cd "d:\Model Cek-Gula"

# Jalankan Jupyter
jupyter notebook

# Atau gunakan JupyterLab
jupyter lab
```

**Di browser:**

1. Buka file `notebook.ipynb`
2. Jalankan setiap cell secara berurutan:
   - **Import Library** → Load & Preprocessing Data
   - **Custom Callback & Layer**
   - **Build & Compile Model**
   - **Training** (akan memakan waktu ~1-3 jam tergantung GPU)
   - **Evaluasi & Visualisasi**
   - **Save Model**

---

### **Opsi 2: Inference dengan Jupyter (Quick Test)**

Jika model sudah terlatih (`best_model.keras` sudah ada):

```bash
jupyter notebook
```

Buka `notebook.ipynb` dan langsung jalankan section:

- **Load Model** → **Inference Model**
- Ubah `image_path='images.jpg'` dengan path gambar Anda

---

### **Opsi 3: REST API Server (FastAPI)**

```bash
# Pastikan model sudah tersimpan (best_model.keras)
cd "d:\Model Cek-Gula"

# Jalankan FastAPI server
uvicorn Main:app --reload --host 127.0.0.1 --port 8000
```

**Output yang muncul:**

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

**Akses API:**

- 🌐 Dokumentasi Interaktif: http://127.0.0.1:8000/docs
- 📖 Alternative Docs: http://127.0.0.1:8000/redoc
- 🏠 Homepage: http://127.0.0.1:8000/

---

## 🔌 Penggunaan API

### Endpoint: POST `/prediksi/`

**Request:**

```bash
curl -X POST "http://127.0.0.1:8000/prediksi/" \
  -H "accept: application/json" \
  -F "file=@path/to/image.jpg" \
  -F "porsi_gram=100"
```

**Parameter:**
| Parameter | Tipe | Deskripsi |
|-----------|------|-----------|
| `file` | File | Gambar jajanan (JPG/PNG) |
| `porsi_gram` | Float | Berat porsi dalam gram (default: 100) |

**Response Sukses (200):**

```json
{
  "status": "success",
  "hasil_analisis": {
    "prediksi_utama": "bakpao",
    "keyakinan": 0.92,
    "porsi_dihitung_gram": 100,
    "informasi_nutrisi": {
      "kalori_kkal": 320.0,
      "karbohidrat_g": 45.0,
      "gula_g": 15.0,
      "protein_g": 8.0,
      "lemak_g": 10.0,
      "indeks_glikemik": 65,
      "beban_glikemik": 22.0,
      "status_risiko": "🟡 SEDANG",
      "pesan_kesehatan": "Konsumsi wajar, perhatikan porsi"
    },
    "prediksi_alternatif": [
      { "jajanan": "bakpao", "probabilitas": 0.92 },
      { "jajanan": "bakwan", "probabilitas": 0.05 },
      { "jajanan": "bolu kukus", "probabilitas": 0.03 }
    ]
  }
}
```
### Endpoint: POST `/saran-kesehatan/`

**Request JSON:**
```json
{
  "nama_jajanan": "bakpao",
  "gula_gram": 15.0,
  "indeks_glikemik": 65,
  "beban_glikemik": 22.0,
  "status_resiko": "🟡 SEDANG"
}
```

**Response Sukses (200):**
```json
{
  "status": "success",
  "saran_kesehatan": "..."
}
```
Catatan: endpoint ini menggunakan Google Gemini API, sehingga `GEMINI_API_KEY` harus diatur di `.env`. Jika key tidak tersedia, endpoint dapat mengembalikan error atau pesan gagal.

### Contoh Penggunaan dengan Python

```python
import requests
from pathlib import Path

# File gambar
image_path = "path/to/jajanan.jpg"

# Kirim request ke API
with open(image_path, "rb") as img:
    response = requests.post(
        "http://127.0.0.1:8000/prediksi/",
        files={"file": img},
        data={"porsi_gram": 100}
    )

# Parse response
hasil = response.json()
print(hasil["hasil_analisis"]["prediksi_utama"])
print(hasil["hasil_analisis"]["informasi_nutrisi"]["gula_g"])
```

### Contoh Penggunaan dengan cURL (Terminal)

```bash
curl -X POST "http://127.0.0.1:8000/prediksi/" \
  -F "file=@C:\path\to\image.jpg" \
  -F "porsi_gram=150"
```

---

## 🧠 Model Details

### Arsitektur Model

```
Input Layer (224, 224, 3)
    ↓
Data Augmentation
  • RandomFlip (horizontal)
  • RandomRotation (0.15)
  • RandomZoom (0.15)
  • RandomContrast (0.1)
    ↓
Custom Layer: NormalisasiGambar
  • Normalisasi: (x / 127.5) - 1.0
    ↓
Base Model: MobileNetV2 (Pretrained ImageNet)
  • Trainable: False (Transfer Learning)
    ↓
GlobalAveragePooling2D
    ↓
Dense (512, ReLU) + BatchNormalization + Dropout(0.5)
    ↓
Dense (num_classes, Softmax)
    ↓
Output: Prediksi Kelas Jajanan
```

### Komponen Custom

#### 1. Custom Layer: `NormalisasiGambar`

```python
class NormalisasiGambar(tf.keras.layers.Layer):
    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        return (inputs / 127.5) - 1.0  # Normalisasi ke range [-1, 1]
```

#### 2. Custom Callback: `TargetAccuracyCallback`

Menghentikan training otomatis ketika akurasi mencapai target (mencegah overfitting)

### Training Configuration

```python
Optimizer    : Adam (learning_rate=0.0005)
Loss         : Categorical Crossentropy
Metrics      : Accuracy, MAE
Batch Size   : 32
Epochs       : 50 (dengan early stopping)
Train/Val    : 80/20 split
```

---

## 📈 Performa Model

### Target Minimum

| Metrik               | Target | Status      |
| -------------------- | ------ | ----------- |
| **Akurasi Validasi** | ≥ 85%  | ✅ tercapai |
| **MAE Validasi**     | ≤ 0.02 | ✅ tercapai |

### Cara Mengecek Performa

**Di Jupyter Notebook:**

```python
# Cek akurasi
best_val_acc = max(history.history['val_accuracy'])
print(f"Best Val Accuracy: {best_val_acc*100:.2f}%")

# Cek MAE
best_val_mae = min(history.history['val_mae'])
print(f"Best Val MAE: {best_val_mae:.4f}")
```

**Visualisasi:**
Setiap training menghasilkan file `train_history.png` dengan 3 grafik:

- Akurasi Training vs Validasi
- Loss Training vs Validasi
- MAE vs Validasi MAE

---

## 🐛 Troubleshooting

### Error: `ModuleNotFoundError: No module named 'tensorflow'`

```bash
pip install tensorflow
```

### Error: `CUDA/GPU tidak terdeteksi`

Model tetap bisa dijalankan di CPU, tapi lebih lambat:

```python
# Di awal notebook/script:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
```

### Error: `FileNotFoundError: best_model.keras`

Model belum ada. Jalankan training terlebih dahulu di Jupyter atau download dari cloud storage.

### Error: `nutrisi_jajanan.csv tidak ditemukan`

API akan berjalan tapi tanpa informasi nutrisi. Tambahkan file CSV ke folder proyek.

### Error: Port 8000 sudah digunakan

```bash
uvicorn Main:app --reload --port 8001  # Sesuaikan port
```

---

## 📝 Notes Penting

⚠️ **Model menggunakan Pretrained MobileNetV2** - Ini adalah transfer learning yang diperbolehkan (base model hanya untuk fitur extraction, tidak langsung digunakan).

⚠️ **Custom Layer diimplementasikan** - `NormalisasiGambar` adalah komponen custom yang dipersyaratkan.

⚠️ **Dataset yang besar** - Pastikan memiliki cukup disk space (~2-5GB) untuk dataset lengkap.

⚠️ **Training memakan waktu** - Pada GPU: ~1 jam, pada CPU: ~3-5 jam.

---

## 📚 Dokumentasi Tambahan

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Jupyter Notebook Guide](https://jupyter.org/)

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan pembelajaran dan riset.

---

**Dibuat untuk Capstone Project Cek-Gula | May 2026**

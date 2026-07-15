# Intrusion Detection System (IDS) menggunakan Arsitektur Hybrid CNN-LSTM

Repositori ini berisi implementasi sistem deteksi intrusi (IDS) berbasis pembelajaran mendalam (*Deep Learning*) menggunakan model gabungan CNN-LSTM untuk dataset **CICIDS2017**. Aplikasi dilengkapi dengan antarmuka grafis (GUI) berbasis **CustomTkinter** untuk mempermudah eksekusi pipeline secara bertahap dari Fase 1 hingga Fase 5.

## 📂 Struktur Repositori

*   `app_tkinter.py`: Antarmuka grafis utama aplikasi (Frontend).
*   `backend_worker_ctk.py`: Pemrosesan pipeline machine learning & deep learning (Backend).
*   `jalankan_aplikasi.bat`: File batch pintasan untuk mengeksekusi aplikasi di lingkungan Windows.
*   `requirements.txt`: Daftar pustaka (dependencies) Python yang diperlukan.

> 📝 **Catatan Infrastruktur & Folder Output**: Folder `outputs/` beserta seluruh sub-direktori visualisasi (`phase1` s.d `phase5`), file model (`.h5`), serta berkas *log execution* **tidak disertakan di dalam repositori Git ini**. Folder tersebut akan dibuat secara otomatis (*auto-generated*) oleh sistem backend pada *root directory* saat aplikasi pertama kali dijalankan.

---

## ⚡ Prasyarat Sistem

Sebelum menjalankan aplikasi, pastikan perangkat Anda telah memenuhi spesifikasi berikut:
*   **Sistem Operasi**: Windows 10 / 11.
*   **Python Version**: Python 3.9 s.d 3.11 (Direkomendasikan).
*   **Dataset**: Unduh berkas CSV asli dari official dataset **CICIDS2017**.

---

## 🚀 Langkah Instalasi & Penggunaan

### 1. Kloning Repositori
Unduh atau kloning repositori ini ke penyimpanan lokal Anda:
```bash
git clone <LINK_REPO_GITHUB_LU>
cd <NAMA_FOLDER_REPO>

2. Pemasangan Pustaka (Dependencies)
Buka terminal (CMD / PowerShell) pada folder proyek, lalu instal semua pustaka yang tertera pada berkas requirements.txt:

pip install -r requirements.txt

3. Konfigurasi Penempatan Dataset
Buat struktur folder berikut di dalam direktori utama proyek, lalu masukkan 8 berkas CSV asli dari dataset CICIDS2017 ke dalamnya:

Dataset/
└── CICIDS2017/
    ├── Monday-WorkingHours.pcap_ISCX.csv
    ├── Tuesday-WorkingHours.pcap_ISCX.csv
    └── ... (dan seterusnya hingga 8 file CSV lengkap)

4. Eksekusi Aplikasi
Untuk menjalankan aplikasi, Anda hanya perlu melakukan klik ganda (double-click) pada file pintasan:
➡️ jalankan_aplikasi.bat

Atau melalui perintah terminal:

python app_tkinter.py
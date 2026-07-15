# Intrusion Detection System (IDS) menggunakan Arsitektur Hybrid CNN-LSTM

Repositori ini berisi implementasi Sistem Deteksi Intrusi (IDS) berbasis pembelajaran mendalam (*Deep Learning*) menggunakan model gabungan CNN-LSTM untuk dataset **CICIDS2017**. Aplikasi ini dilengkapi dengan Antarmuka Grafis Pengguna (GUI) berbasis **CustomTkinter** untuk mempermudah eksekusi seluruh *pipeline* secara bertahap, mulai dari Fase 1 hingga Fase 5.

## 📂 Struktur Repositori

*   `app_tkinter.py`: Antarmuka grafis utama aplikasi (Frontend / User Interface).
*   `backend_worker_ctk.py`: Pemrosesan utama komponen machine learning & deep learning (Backend Worker).
*   `jalankan_aplikasi.bat`: Berkas pintasan skrip (*batch file*) untuk mempermudah eksekusi aplikasi di lingkungan sistem operasi Windows.
*   `requirements.txt`: Daftar pustaka dependensi Python yang diperlukan oleh sistem.

> 📝 **Catatan Pustaka & Folder Output**: Folder `outputs/` beserta seluruh sub-direktori visualisasi (`phase1` s.d `phase5`), berkas model final (`.h5`), serta dokumen catatan riwayat eksekusi (*log files*) tidak disertakan di dalam repositori ini. Seluruh direktori tersebut akan dibuat secara otomatis (*auto-generated*) oleh sistem backend pada *root directory* saat aplikasi pertama kali dijalankan.

---

## ⚡ Prasyarat Sistem

Sebelum menjalankan aplikasi, pastikan perangkat keras dan lingkungan perangkat lunak telah memenuhi spesifikasi berikut:
*   **Sistem Operasi**: Windows 10 / 11.
*   **Versi Python**: Python 3.9 s.d 3.11 (Direkomendasikan).
*   **Dataset**: Berkas CSV asli dari official dataset **CICIDS2017**.

---

## 🚀 Langkah Instalasi & Penggunaan

### 1. Kloning Repositori
Unduh atau kloning repositori ini ke dalam penyimpanan lokal perangkat:
git clone [https://github.com/firgiansyahw/ids_cnn_lstm.git](https://github.com/firgiansyahw/ids_cnn_lstm.git)
cd ids_cnn_lstm

2. Pemasangan Pustaka Dependensi
Buka terminal pilihan Anda (Command Prompt / PowerShell) pada folder utama proyek, lalu jalankan perintah berikut untuk menginstal seluruh pustaka yang tertera pada berkas requirements.txt:
pip install -r requirements.txt

4. Konfigurasi dan Penempatan Dataset
Buat sebuah direktori baru bernama Dataset/ di dalam folder utama proyek, kemudian susun struktur folder internalnya seperti di bawah ini untuk meletakkan 8 berkas CSV asli dari dataset CICIDS2017:  

Plaintext
Dataset/

└── CICIDS2017/
    ├── Monday-WorkingHours.pcap_ISCX.csv
    ├── Tuesday-WorkingHours.pcap_ISCX.csv
    ├── Wednesday-workingHours.pcap_ISCX.csv
    ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
    ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
    ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
    └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    
4. Mengeksekusi Aplikasi
Untuk menjalankan aplikasi secara langsung di lingkungan Windows, pengguna dapat melakukan klik ganda (double-click) pada berkas pintasan:
➡️ jalankan_aplikasi.bat

Sebagai alternatif, aplikasi juga dapat dijalankan secara manual melalui perintah terminal berikut:
python app_tkinter.py

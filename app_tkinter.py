# -*- coding: utf-8 -*-
# CATATAN REVISI:
# main pipeline: Phase 1 sampai Phase 5.

"""
app_tkinter.py
UI berbasis CustomTkinter sebagai pengganti app.py (PyQt6)
Requires: pip install customtkinter pillow
"""


# =============================================================================
# CATATAN SIDANG
# =============================================================================
# File ini berperan sebagai FRONTEND / USER INTERFACE aplikasi IDS.
# File ini tidak berisi algoritma machine learning utama.
# Tugasnya adalah:
# 1. Menampilkan tab Phase 1 sampai Phase 5.
# 2. Menyediakan tombol START untuk setiap fase.
# 3. Menjalankan backend worker di thread terpisah.
# 4. Menampilkan progress bar dan process log.
# 5. Membaca file PNG dari folder outputs/phase-x lalu menampilkannya di gallery.
#
# Hubungan file:
# app_tkinter.py        = tampilan aplikasi
# backend_worker_ctk.py = mesin proses ML yang dipanggil oleh UI
# =============================================================================

import os
import sys
import threading
import queue
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk

# Anti-Deadlock untuk OpenMP.
# Beberapa library ML seperti TensorFlow, NumPy, atau scikit-learn bisa memanggil
# runtime OpenMP. Baris ini membantu mencegah konflik runtime pada Windows.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Tema global CustomTkinter ──────────────────────────────────────────────────
# Mengatur tampilan aplikasi: dark mode dan tema warna biru.
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ==============================================================================
# ADAPTER: Menghubungkan IDSWorker ke Tkinter
# ==============================================================================
# WorkerAdapter adalah jembatan antara UI dan backend.
#
# Kenapa perlu adapter?
# - Backend menjalankan proses berat seperti training dan evaluasi.
# - Jika proses berat dijalankan langsung pada main thread Tkinter, UI akan freeze.
# - Karena itu, backend dijalankan pada threading.Thread.
# - Pesan log/progress dikirim ke UI melalui callback dan queue.
# ==============================================================================
class WorkerAdapter:
    """
    Membungkus IDSWorker supaya bisa dipakai tanpa PyQt6.
    Sinyal diganti dengan queue + callback.
    """
    def __init__(self, on_log, on_progress, on_finished, on_error):
        self.on_log      = on_log
        self.on_progress = on_progress
        self.on_finished = on_finished
        self.on_error    = on_error
        self._thread     = None

    # Mengganti signal bawaan worker dengan callback dari UI.
    # Jadi saat backend memanggil log_signal.emit(), pesan masuk ke queue UI.
    def _patch_worker(self, worker):
        """Ganti pyqtSignal.emit dengan fungsi biasa."""
        worker.log_signal      = type('S', (), {'emit': lambda _, m: self.on_log(m)})()
        worker.progress_signal = type('S', (), {'emit': lambda _, v: self.on_progress(v)})()
        worker.finished_signal = type('S', (), {'emit': lambda _: self.on_finished()})()
        worker.error_signal    = type('S', (), {'emit': lambda _, m: self.on_error(m)})()
        return worker

    # Menjalankan salah satu fase backend berdasarkan tombol yang ditekan user.
    # phase=1 memanggil run_phase1(), phase=2 memanggil run_phase2(), dst.
    def run(self, phase, model_path=None, data_path=None):
        # Import IDSWorker dari backend.
        # Import dilakukan di dalam function agar aplikasi tidak langsung crash
        # saat file UI dibuka, dan agar dependency backend baru dimuat saat dibutuhkan.
        # Import di sini supaya tidak crash kalau PyQt6 tidak ada
        # backend_worker menggunakan QObject tapi kita patch sinyalnya
        try:
            from backend_worker_ctk import IDSWorker
        except ImportError:
            from backend_worker import IDSWorker

        worker = IDSWorker()
        worker = self._patch_worker(worker)

        # Dictionary ini menghubungkan nomor phase dengan method backend.
        # Contoh: phase 4 -> worker.run_phase4
        phase_methods = {
            1: worker.run_phase1,
            2: worker.run_phase2,
            3: worker.run_phase3,
            4: worker.run_phase4,
            5: worker.run_phase5,
        }

        # Proses backend dijalankan pada thread terpisah agar UI tetap responsif.
        self._thread = threading.Thread(target=phase_methods[phase], daemon=True)
        self._thread.start()

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()


# ==============================================================================
# MAIN WINDOW
# ==============================================================================
# MainWindow adalah class utama untuk tampilan aplikasi.
# Di dalamnya terdapat header, tab fase, tombol START, log box, progress bar,
# dan gallery untuk menampilkan output PNG.
# ==============================================================================
class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("IDS System - Firgi (CNN-LSTM)")
        self.geometry("1280x860")
        self.minsize(1000, 700)

        # Queue untuk komunikasi thread → UI.
        # Tkinter tidak thread-safe, jadi thread backend tidak boleh langsung
        # mengubah widget UI. Pesan dari backend masuk ke queue, lalu dibaca
        # oleh main thread melalui _poll_queue().
        self._queue = queue.Queue()
        self._adapter = None
        self._img_refs = {}   # Simpan referensi gambar agar tidak di-GC

        self._build_ui()
        self._poll_queue()   # Mulai polling queue setiap 100ms

    # ──────────────────────────────────────────────────────────────────────────
    # BUILD UI
    # Bagian ini membangun semua komponen tampilan aplikasi.
    # ──────────────────────────────────────────────────────────────────────────
    # Membuat header, subtitle, tab fase, log area, dan progress bar.
    def _build_ui(self):
        # Header aplikasi:
        # Menampilkan judul sistem pada bagian atas window.
        # ── Header ────────────────────────────────────────────────────────────
        header = ctk.CTkLabel(
            self,
            text="Intrusion Detection System (CNN-LSTM)",
            font=ctk.CTkFont(family="Arial", size=20, weight="bold")
        )
        header.pack(pady=(16, 4))

        subtitle = ctk.CTkLabel(
            self,
            text="CICIDS2017 | RF · DT · KNN · CNN-LSTM Comparison",
            font=ctk.CTkFont(size=13),
            text_color="gray"
        )
        subtitle.pack(pady=(0, 10))

        # Tab view:
        # Setiap fase pipeline dibuat sebagai tab terpisah agar user bisa menjalankan
        # proses secara bertahap: Loading, Cleaning, Feature Engineering, Modeling, Evaluation.
        # ── Tab view ──────────────────────────────────────────────────────────
        self.tabview = ctk.CTkTabview(self, height=480)
        self.tabview.pack(fill="x", padx=20, pady=(0, 8))

        # Daftar Phase 1 sampai Phase 5.
        # Tiap item berisi:
        # (nama tab, judul fase, deskripsi fase, nomor fase)
        phases = [
            ("1. Loading",    "Fase 1: Data Loading",        "Menggabungkan dataset CSV dari folder Dataset/CICIDS2017.", 1),
            ("2. Cleaning",   "Fase 2: Preprocessing",       "Drop duplikat, handle missing/inf, attack mapping.", 2),
            ("3. Feature Eng","Fase 3: Feature Engineering", "Memory optimization, PCA, SMOTE balancing.", 3),
            ("4. Modeling",   "Fase 4: Training",            "Latih RF, DT, KNN, dan CNN-LSTM final.", 4),
            ("5. Eval",       "Fase 5: Evaluation",          "Confusion matrix, classification report, perbandingan akurasi.", 5),
        ]
        self._gallery_frames = {}

        for tab_name, title, desc, phase_id in phases:
            self.tabview.add(tab_name)
            self._build_phase_tab(self.tabview.tab(tab_name), title, desc, phase_id)

                        # dan model CNN-LSTM .h5 hasil training.

        # Log area:
        # Menampilkan pesan proses dari backend, misalnya data berhasil dibaca,
        # model selesai dilatih, atau error yang terjadi.
        # ── Log area ──────────────────────────────────────────────────────────
        log_label = ctk.CTkLabel(self, text="Process Log:", anchor="w",
                                 font=ctk.CTkFont(size=12))
        log_label.pack(fill="x", padx=20)

        self.log_box = ctk.CTkTextbox(
            self, height=130,
            font=ctk.CTkFont(family="Consolas", size=11),
            text_color="#00ff00",
            fg_color="#1e1e1e"
        )
        self.log_box.pack(fill="x", padx=20, pady=(0, 6))
        self.log_box.configure(state="disabled")

        # Progress bar:
        # Menampilkan persentase progres dari backend.
        # Nilainya dikirim dari backend melalui progress_signal.
        # ── Progress bar ──────────────────────────────────────────────────────
        self.progress_var = ctk.DoubleVar(value=0)
        self.progress_bar = ctk.CTkProgressBar(self, variable=self.progress_var,
                                               height=16)
        self.progress_bar.pack(fill="x", padx=20, pady=(0, 12))
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(self, text="0%", font=ctk.CTkFont(size=11))
        self.progress_label.pack()

    # Membuat tampilan untuk tiap tab Phase 1 sampai Phase 5.
    # Setiap tab memiliki judul, deskripsi, tombol START, dan gallery output.
    def _build_phase_tab(self, parent, title, desc, phase_id):
        # Title + desc
        ctk.CTkLabel(parent, text=title,
                     font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=12, pady=(10, 2))
        ctk.CTkLabel(parent, text=desc,
                     font=ctk.CTkFont(size=12), text_color="gray",
                     wraplength=900, justify="left").pack(anchor="w", padx=12, pady=(0, 8))

        btn = ctk.CTkButton(
            parent,
            text=f"▶  START {title.upper()}",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38,
            command=lambda p=phase_id: self._start_process(p)
        )
        btn.pack(padx=12, pady=(0, 8))

        # Gallery label
        ctk.CTkLabel(parent, text="Visualisasi Output:",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=12)

        # Scrollable gallery
        gallery = ctk.CTkScrollableFrame(parent, height=260)
        gallery.pack(fill="both", expand=True, padx=12, pady=(4, 8))
        self._gallery_frames[phase_id] = gallery


    # ──────────────────────────────────────────────────────────────────────────
    # THREAD-SAFE QUEUE POLLING
    # Fungsi polling ini membaca pesan dari queue setiap 100 ms.
    # Tujuannya agar update UI tetap dilakukan dari main thread Tkinter.
    # Tkinter tidak thread-safe, semua update UI harus dari main thread.
    # Worker menaruh pesan di queue, main thread memproses via after().
    # ──────────────────────────────────────────────────────────────────────────
    # Membaca pesan dari queue:
    # - log      -> ditampilkan ke log box
    # - progress -> update progress bar
    # - finished -> load gallery dan tampilkan info selesai
    # - error    -> tampilkan pesan error
    def _poll_queue(self):
        try:
            while True:
                msg = self._queue.get_nowait()
                msg_type = msg[0]

                if msg_type == "log":
                    self._append_log(msg[1])
                elif msg_type == "progress":
                    val = msg[1] / 100.0
                    self.progress_bar.set(val)
                    self.progress_label.configure(text=f"{msg[1]}%")
                elif msg_type == "finished":
                    phase = msg[1]
                    self._on_finished(phase)
                elif msg_type == "error":
                    self._on_error(msg[1])

        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_queue)

    # ──────────────────────────────────────────────────────────────────────────
    # CALLBACKS
    # Callback ini diberikan ke WorkerAdapter.
    # Ketika backend mengirim log/progress, callback memasukkan pesan ke queue.
    # ──────────────────────────────────────────────────────────────────────────
    # Membuat 4 callback untuk satu phase:
    # on_log, on_progress, on_finished, dan on_error.
    def _make_callbacks(self, phase):
        q = self._queue
        return (
            lambda msg: q.put(("log",      msg)),
            lambda val: q.put(("progress", val)),
            lambda:     q.put(("finished", phase)),
            lambda msg: q.put(("error",    msg)),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # ACTIONS
    # Bagian ini berisi aksi yang terjadi ketika user menekan tombol di UI.
    # ──────────────────────────────────────────────────────────────────────────
    # Dipanggil saat user menekan tombol START pada Phase 1 sampai Phase 5.
    # Fungsi ini membuat WorkerAdapter dan menjalankan backend sesuai nomor phase.
    def _start_process(self, phase):
        if self._adapter and self._adapter.is_running():
            messagebox.showwarning("Tunggu", "Proses sedang berjalan!")
            return

        self.progress_bar.set(0)
        self.progress_label.configure(text="0%")
        self._append_log(f"\n{'─'*50}\n▶ Memulai Fase {phase}\n{'─'*50}")
        self._clear_gallery(phase)

        on_log, on_prog, on_fin, on_err = self._make_callbacks(phase)
        self._adapter = WorkerAdapter(on_log, on_prog, on_fin, on_err)
        self._adapter.run(phase)


    # ──────────────────────────────────────────────────────────────────────────
    # FINISHED / ERROR
    # Bagian ini menangani kondisi ketika backend selesai atau error.
    # ──────────────────────────────────────────────────────────────────────────
    # Dipanggil ketika backend selesai menjalankan suatu phase.
    # UI kemudian memuat visualisasi output dari folder outputs/phaseX.
    def _on_finished(self, phase):
        self._append_log(f"✓ Fase {phase} selesai!")
        self._load_gallery(phase)
        messagebox.showinfo("Sukses", f"Fase {phase} selesai!")

    # Dipanggil ketika backend mengirim pesan error.
    # Error ditampilkan di log box dan messagebox.
    def _on_error(self, msg):
        self._append_log(f"[ERROR] {msg}")
        messagebox.showerror("Error", msg)

    # ──────────────────────────────────────────────────────────────────────────
    # LOG HELPER
    # Fungsi kecil untuk menambahkan teks ke log box.
    # ──────────────────────────────────────────────────────────────────────────
    # Menambahkan pesan ke log box.
    # Textbox dibuat normal sementara, pesan dimasukkan, lalu dikunci kembali.
    def _append_log(self, msg):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    # ──────────────────────────────────────────────────────────────────────────
    # GALLERY
    # Bagian ini menampilkan file PNG yang dihasilkan setiap fase.
    # ──────────────────────────────────────────────────────────────────────────
    # Menghapus gambar lama dari gallery sebelum phase dijalankan ulang.
    def _clear_gallery(self, phase):
        frame = self._gallery_frames.get(phase)
        if not frame:
            return
        for widget in frame.winfo_children():
            widget.destroy()
        self._img_refs.pop(phase, None)

    # Membaca semua file PNG di outputs/phaseX lalu menampilkannya di UI.
    # Gambar disimpan sebagai CTkImage dan referensinya ditahan agar tidak terhapus
    # oleh garbage collector Python.
    def _load_gallery(self, phase):
        # Path relatif ke lokasi script, bukan working directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        folder = os.path.join(script_dir, "outputs", f"phase{phase}")
        frame  = self._gallery_frames.get(phase)
        if not frame or not os.path.exists(folder):
            return

        images = sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])
        if not images:
            ctk.CTkLabel(frame, text="Tidak ada gambar output.",
                         text_color="gray").grid(row=0, column=0, padx=8, pady=8)
            return

        self._img_refs[phase] = []
        col_max = 3
        for idx, img_file in enumerate(images):
            img_path = os.path.join(folder, img_file)
            row = idx // col_max
            col = idx % col_max

            card = ctk.CTkFrame(frame, corner_radius=8)
            card.grid(row=row, column=col, padx=8, pady=8, sticky="nw")

            # Nama file
            ctk.CTkLabel(card, text=img_file, font=ctk.CTkFont(size=10),
                         text_color="gray", wraplength=280).pack(pady=(6, 2))

            # Gambar
            try:
                pil_img  = Image.open(img_path)
                pil_img.thumbnail((320, 240), Image.LANCZOS)
                ctk_img  = ctk.CTkImage(light_image=pil_img, dark_image=pil_img,
                                        size=pil_img.size)
                lbl_img  = ctk.CTkLabel(card, image=ctk_img, text="")
                lbl_img.pack(padx=8, pady=(0, 8))
                self._img_refs[phase].append(ctk_img)   # cegah GC
            except Exception as e:
                ctk.CTkLabel(card, text=f"[Gagal load: {e}]",
                             text_color="red").pack(padx=8, pady=8)


# ==============================================================================
# ENTRY POINT
# ==============================================================================
# Saat file app_tkinter.py dijalankan, MainWindow dibuat dan event loop dimulai.
# Dari titik ini aplikasi menunggu interaksi user melalui tombol-tombol phase.
# ==============================================================================
if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()
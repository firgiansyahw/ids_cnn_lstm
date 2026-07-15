@echo off
title Menjalankan Aplikasi IDS CNN-LSTM - Firgi
echo =====================================================================
echo            INTRUSION DETECTION SYSTEM (CNN-LSTM MODEL)
echo =====================================================================
echo.
echo [INFO] Memeriksa komponen dan menjalankan Antarmuka Grafis...
echo [INFO] Folder 'outputs/' akan dibuat otomatis jika belum tersedia.
echo.
python app_tkinter.py
echo.
echo =====================================================================
echo [INFO] Aplikasi telah ditutup dengan aman.
echo =====================================================================
pause
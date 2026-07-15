
# Phase 1: data loading
# Phase 2: data cleaning/preprocessing
# Phase 3: feature engineering, PCA, SMOTE
# Phase 4: model training
# Phase 5: model evaluation
#
# File ini dibuat agar pipeline berjalan linear di script
# eksperimen ids_cnn_lstm.py dapat dijalankan per fase dari tampilan UI.
#
# Prinsip sederhananya:
# app_tkinter.py = tampilan / tombol / log / progress bar
# backend_worker_ctk.py = mesin proses yang dipanggil oleh tombol tersebut
# =============================================================================

import os
import json
import pickle
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd


# ── Path config ───────────────────────────────────────────────────────────────
# Bagian ini menentukan lokasi dasar project, folder output, dan folder dataset.
# BASE_DIR memakai lokasi file ini agar path tetap relatif terhadap folder aplikasi.
# Dengan cara ini, aplikasi lebih mudah dijalankan melalui batch file atau terminal.
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs")
DATASET_DIR    = os.path.join(BASE_DIR, "Dataset", "CICIDS2017")

# Daftar 8 file CSV CICIDS2017 yang dipakai sebagai data utama training.
# Phase 1 akan membaca semua file ini lalu menggabungkannya menjadi satu dataset.
DATASET_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

# ATTACK_MAP digunakan untuk menyederhanakan label asli dataset menjadi kelas umum yang dipakai oleh model, misalnya:
# - DoS Hulk, DoS GoldenEye, DoS Slowloris -> DoS
# - FTP-Patator, SSH-Patator -> Brute Force
# - beberapa varian Web Attack -> Web Attack
ATTACK_MAP = {
    # ── CICIDS2017 ─────────────────────────────────────────────────────────────
    'BENIGN': 'BENIGN', 'DDoS': 'DDoS',
    'DoS Hulk': 'DoS', 'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS', 'DoS Slowhttptest': 'DoS',
    'PortScan': 'Port Scan',
    'FTP-Patator': 'Brute Force', 'SSH-Patator': 'Brute Force',
    'Bot': 'Bot',
    # Web Attack — semua varian encoding
    'Web Attack \x96 Brute Force':   'Web Attack',
    'Web Attack \x96 XSS':           'Web Attack',
    'Web Attack \x96 Sql Injection':  'Web Attack',
    'Web Attack \u2013 Brute Force':  'Web Attack',
    'Web Attack \u2013 XSS':          'Web Attack',
    'Web Attack \u2013 Sql Injection': 'Web Attack',
    'Web Attack \u2014 Brute Force':  'Web Attack',
    'Web Attack \u2014 XSS':          'Web Attack',
    'Web Attack \u2014 Sql Injection': 'Web Attack',
    'Web Attack \ufffd Brute Force':  'Web Attack',
    'Web Attack \ufffd XSS':          'Web Attack',
    'Web Attack \ufffd Sql Injection': 'Web Attack',
}


# ── Dummy signal (akan di-override oleh WorkerAdapter) ───────────────────────
# _Signal adalah pengganti sederhana dari pyqtSignal.
# Karena aplikasi memakai Tkinter, sinyal ini nantinya diganti oleh callback
# dari app_tkinter.py agar backend bisa mengirim log, progress, error, dan status selesai.
class _Signal:
    def emit(self, *args): pass

# ── Helper I/O ────────────────────────────────────────────────────────────────
# Fungsi-fungsi helper ini dipakai untuk menyimpan output:
# - PNG untuk visualisasi
# - JSON untuk ringkasan hasil
# - PKL untuk model atau objek preprocessing seperti scaler dan PCA
def _save_plot(filename, phase, dpi=150):
    import matplotlib.pyplot as plt
    folder = os.path.join(OUTPUT_DIR, f"phase{phase}")
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, f"{filename}.png"), dpi=dpi, bbox_inches='tight')
    plt.close()

def _save_json(data, filename, phase):
    folder = os.path.join(OUTPUT_DIR, f"phase{phase}")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"{filename}.json"), 'w') as f:
        json.dump(data, f, indent=2)

def _save_pickle(obj, filename, phase):
    folder = os.path.join(OUTPUT_DIR, f"phase{phase}")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"{filename}.pkl"), 'wb') as f:
        pickle.dump(obj, f)

def _load_pickle(filename, phase):
    with open(os.path.join(OUTPUT_DIR, f"phase{phase}", f"{filename}.pkl"), 'rb') as f:
        return pickle.load(f)

# ==============================================================================
# WORKER CLASS (tanpa QObject)
# ==============================================================================
class IDSWorker:
    def __init__(self):
        self.log_signal      = _Signal()
        self.progress_signal = _Signal()
        self.finished_signal = _Signal()
        self.error_signal    = _Signal()
        self.val_model_path  = None
        self.val_data_path   = None

        # Run ID dipakai agar setiap eksekusi aplikasi memiliki jejak log sendiri.
        # Format timestamp dibuat aman untuk nama file Windows.
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_phase = None
        self._log_dir = os.path.join(OUTPUT_DIR, "logs")
        self._run_log_dir = os.path.join(self._log_dir, "runs")
        self._full_log_path = os.path.join(self._run_log_dir, f"{self.run_id}_full_log.txt")
        self._current_log_path = os.path.join(self._log_dir, "current_run_full_log.txt")

    def _ensure_log_dirs(self):
        os.makedirs(self._log_dir, exist_ok=True)
        os.makedirs(self._run_log_dir, exist_ok=True)

    def _set_phase(self, phase):
        # Menandai phase aktif agar log bisa ditulis juga ke file phase spesifik.
        self._current_phase = int(phase)
        self._ensure_log_dirs()
        self._write_log_file_only(f"===== PHASE {phase} START =====")

    def _write_log_file_only(self, msg):
        # Menulis log ke file tanpa mengirimkannya ke UI.
        # Cocok untuk detail yang panjang seperti epoch training CNN-LSTM.
        self._ensure_log_dirs()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}\n"

        # Full log per run.
        with open(self._full_log_path, "a", encoding="utf-8") as f:
            f.write(line)

        # Full log terbaru yang selalu ditimpa per session baru.
        with open(self._current_log_path, "a", encoding="utf-8") as f:
            f.write(line)

        # Log khusus phase aktif.
        if self._current_phase is not None:
            phase_path = os.path.join(
                self._run_log_dir,
                f"{self.run_id}_phase{self._current_phase}_log.txt"
            )
            with open(phase_path, "a", encoding="utf-8") as f:
                f.write(line)

    # Fungsi kecil untuk mengirim pesan log ke UI dan menyimpannya ke file log.
    def _log(self, msg):
        msg = str(msg)
        self.log_signal.emit(msg)
        self._write_log_file_only(msg)

    # Fungsi kecil untuk mengirim nilai progress bar ke UI.
    def _prog(self, val):  self.progress_signal.emit(int(val))

    # Dipanggil ketika sebuah phase selesai.
    def _finish(self):
        self._write_log_file_only(f"===== PHASE {self._current_phase} FINISHED =====")
        self.finished_signal.emit()

    # Dipanggil ketika terjadi error agar UI dapat menampilkan pesan error.
    def _error(self, msg):
        msg = str(msg)
        self._write_log_file_only(f"[ERROR] {msg}")
        self.error_signal.emit(msg)

    # Membuat folder outputs/phase1 sampai outputs/phase5.
    # Folder ini digunakan untuk menyimpan hasil tiap fase secara terstruktur.
    def _make_dirs(self):
        for i in range(1, 6):
            os.makedirs(os.path.join(OUTPUT_DIR, f"phase{i}"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "phase4", "models"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "logs"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "logs", "runs"), exist_ok=True)

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    # PHASE 1: DATA LOADING
    # Fungsi:
    # 1. Membaca 8 file CSV CICIDS2017.
    # 2. Menggabungkan semuanya menjadi satu dataframe.
    # 3. Membersihkan whitespace pada nama kolom.
    # 4. Menyimpan hasil gabungan ke outputs/phase1/merged_data.parquet.
    #
    # File parquet dipakai agar fase berikutnya lebih cepat membaca data
    # dibanding harus membaca ulang semua CSV.
    def run_phase1(self):
        try:
            self._set_phase(1)
            import matplotlib; matplotlib.use('Agg')
            self._make_dirs()
            self._log("=" * 60)
            self._log("PHASE 1: DATA LOADING")
            self._log("=" * 60)
            self._prog(5)

            data_list = []
            for i, fname in enumerate(DATASET_FILES):
                fpath = os.path.join(DATASET_DIR, fname)
                self._log(f"Loading: {fname}")
                df = pd.read_csv(fpath, encoding='latin-1')
                data_list.append(df)
                self._log(f"  Data{i+1} → {df.shape[0]} rows, {df.shape[1]} cols")
                self._prog(5 + int((i + 1) / len(DATASET_FILES) * 55))

            self._log("Menggabungkan dataset...")
            data = pd.concat(data_list).reset_index(drop=True)
            for d in data_list: del d
            data.rename(columns={c: c.strip() for c in data.columns}, inplace=True)
            self._prog(75)

            data.to_parquet(os.path.join(OUTPUT_DIR, "phase1", "merged_data.parquet"), index=False)
            self._log(f"✓ Saved: phase1/merged_data.parquet ({data.shape[0]} rows)")
            _save_json({'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'run_id': self.run_id,
                        'full_log': self._full_log_path,
                        'current_log': self._current_log_path,
                        'total_rows': int(data.shape[0]), 'total_cols': int(data.shape[1])},
                       'data_loading_summary', 1)
            self._prog(100)
            self._log("✓ Phase 1 selesai!")
            self._finish()
        except Exception:
            self._error(f"Phase 1 Error:\n{traceback.format_exc()}")

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    # PHASE 2: DATA CLEANING / PREPROCESSING
    # Fungsi:
    # 1. Membaca merged_data.parquet dari Phase 1.
    # 2. Menghapus data duplikat.
    # 3. Mengganti nilai infinity menjadi NaN.
    # 4. Mengisi missing value pada Flow Bytes/s dan Flow Packets/s dengan median.
    # 5. Melakukan mapping label attack ke kelas umum.
    # 6. Menyimpan data bersih ke outputs/phase2/clean_data.parquet.
    def run_phase2(self):
        try:
            self._set_phase(2)
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            import missingno as msno
            sns.set(style='darkgrid')

            self._log("=" * 60); self._log("PHASE 2: DATA CLEANING"); self._log("=" * 60)
            self._prog(5)

            data = pd.read_parquet(os.path.join(OUTPUT_DIR, "phase1", "merged_data.parquet"))
            self._prog(15)

            n_dup = data.duplicated().sum()
            data.drop_duplicates(inplace=True)
            self._log(f"Drop duplikat: {n_dup} → shape {data.shape}")

            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            for col in ['Flow Bytes/s', 'Flow Packets/s']:
                if col in data.columns:
                    data[col] = data[col].fillna(data[col].median())
            self._prog(30)

            # Normalisasi label:
            # - strip() untuk menghapus spasi berlebih
            # - replace karakter rusak akibat encoding
            # Tujuannya agar label Web Attack dan label lain bisa ter-map dengan benar.
            # Normalize label: strip + fix encoding mojibake
            # latin-1 read seharusnya sudah utuh, tapi jika parquet menyimpan
            # sebagai UTF-8, en-dash bisa jadi \ufffd atau ï¿½
            data['Label'] = data['Label'].astype(str).str.strip() \
                .str.replace('\ufffd', '\x96', regex=False) \
                .str.replace('ï¿½', '\x96', regex=False)

            data['Attack Type'] = data['Label'].map(ATTACK_MAP)
            n_unmapped = data['Attack Type'].isna().sum()
            if n_unmapped:
                unmapped_labels = data.loc[data['Attack Type'].isna(), 'Label'].value_counts()
                self._log(f"Drop {n_unmapped} baris label tidak ter-mapping:")
                for lbl, cnt in unmapped_labels.items():
                    self._log(f"  {repr(lbl)}: {cnt}")
                data.dropna(subset=['Attack Type'], inplace=True)
            data.drop('Label', axis=1, inplace=True)
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            data['Attack Number'] = le.fit_transform(data['Attack Type'])
            self._log(f"Kelas setelah mapping: {sorted(data['Attack Type'].unique())}")
            self._prog(45)

            # Membuat visualisasi EDA sederhana.
            # Output PNG akan disimpan ke folder phase2 dan ditampilkan di gallery UI.
            # Plots
            self._log("Membuat visualisasi...")
            missing_cols = [c for c in data.columns if data[c].isna().any()]
            if missing_cols:
                fig, ax = plt.subplots(figsize=(4, 6))
                msno.bar(data[missing_cols], ax=ax, fontsize=10)
                _save_plot('missing_values_chart', 2)

            sample = data.sample(n=min(50000, len(data)), random_state=0)
            corr = sample.corr(numeric_only=True).round(2)
            fig, ax = plt.subplots(figsize=(20, 20))
            sns.heatmap(corr, cmap='coolwarm', annot=False, linewidth=0.3, ax=ax)
            plt.title('Correlation Matrix')
            _save_plot('correlation_matrix', 2, dpi=80)
            self._prog(65)

            ac = data['Attack Type'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=ac.values, y=ac.index, ax=ax, palette='pastel', hue=ac.index, legend=False)
            ax.set_title('Attack Type Distribution')
            _save_plot('attack_types_count', 2)

            # Pie chart — exclude BENIGN supaya distribusi serangan terbaca jelas
            ac_attacks = data[data['Attack Type'] != 'BENIGN']['Attack Type'].value_counts()
            threshold  = 0.005
            pct        = ac_attacks / ac_attacks.sum()
            small      = pct[pct < threshold].index.tolist()
            if small:
                ac_attacks['Other'] = ac_attacks[small].sum()
                ac_attacks.drop(small, inplace=True)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(ac_attacks.values, labels=ac_attacks.index, autopct='%1.1f%%')
            ax.set_title('Distribution of Attack Types')
            ax.legend(ac_attacks.index, loc='best')
            _save_plot('attack_distribution_pie', 2)
            self._prog(80)

            data.to_parquet(os.path.join(OUTPUT_DIR, "phase2", "clean_data.parquet"), index=False)
            _save_json({'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'duplicates_removed': int(n_dup),
                        'rows_after_clean': int(data.shape[0])},
                       'preprocessing_summary', 2)
            self._log("✓ Phase 2 selesai!")
            self._prog(100)
            self._finish()
        except Exception:
            self._error(f"Phase 2 Error:\n{traceback.format_exc()}")

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    # PHASE 3: FEATURE ENGINEERING
    # Fungsi:
    # 1. Membaca clean_data.parquet dari Phase 2.
    # 2. Melakukan downcasting untuk menghemat RAM.
    # 3. Menghapus fitur zero-variance.
    # 4. Melakukan StandardScaler.
    # 5. Melakukan Incremental PCA.
    # 6. Melakukan balancing dataset menggunakan SMOTE.
    # 7. Membagi data menjadi train dan test.
    # 8. Menyimpan artifact preprocessing dan data train-test.
    def run_phase3(self):
        try:
            self._set_phase(3)
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from sklearn.decomposition import IncrementalPCA
            from sklearn.impute import SimpleImputer
            from sklearn.model_selection import train_test_split
            from imblearn.over_sampling import SMOTE

            self._log("=" * 60); self._log("PHASE 3: FEATURE ENGINEERING"); self._log("=" * 60)
            self._prog(5)

            data = pd.read_parquet(os.path.join(OUTPUT_DIR, "phase2", "clean_data.parquet"))
            self._prog(10)

            # Downcasting:
            # Mengubah float64 -> float32 dan int64 -> int32 jika aman.
            # Tujuannya mengurangi penggunaan memori karena CICIDS2017 sangat besar.
            # Downcast
            old_mem = data.memory_usage().sum() / 1024**2
            for col in data.select_dtypes('float64').columns:
                mn, mx = data[col].min(), data[col].max()
                if mn > np.finfo(np.float32).min and mx < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
            for col in data.select_dtypes('int64').columns:
                mn, mx = data[col].min(), data[col].max()
                if mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
            new_mem = data.memory_usage().sum() / 1024**2
            self._log(f"Memory: {old_mem:.1f} MB → {new_mem:.1f} MB ({(1-new_mem/old_mem)*100:.1f}% reduction)")
            self._prog(20)

            # Drop zero-variance:
            # Fitur yang nilainya sama di semua baris tidak membantu klasifikasi,
            # sehingga dihapus sebelum scaling dan PCA.
            # Drop zero-variance
            num_uniq = data.select_dtypes(include=np.number).nunique()
            drop_cols = num_uniq[num_uniq == 1].index.tolist()
            data = data[[c for c in data.columns if c not in drop_cols]]

            features = data.drop(['Attack Type', 'Attack Number'], axis=1, errors='ignore')
            attacks  = data['Attack Type']

            # StandardScaler:
            # Menstandarkan fitur agar memiliki skala yang sebanding.
            # Ini penting sebelum PCA dan model berbasis jarak/deep learning.
            scaler  = StandardScaler()
            scaled  = scaler.fit_transform(features)
            imputer = SimpleImputer(strategy='median')
            scaled  = imputer.fit_transform(scaled)
            self._prog(35)

            # Incremental PCA:
            # Mengurangi dimensi fitur menjadi setengah dari jumlah fitur awal.
            # IncrementalPCA dipilih karena lebih cocok untuk dataset besar.
            n_comp = len(features.columns) // 2 #fitur awal 71 diatur jadi 35 karena dibagi 2
            ipca   = IncrementalPCA(n_components=n_comp, batch_size=500)
            for batch in np.array_split(scaled, max(1, len(features) // 500)):
                ipca.partial_fit(batch)
            var = sum(ipca.explained_variance_ratio_)
            self._log(f"PCA: {n_comp} components, variance retained: {var:.2%}")
            transformed = ipca.transform(scaled)
            self._prog(55)

            new_data = pd.DataFrame(transformed, columns=[f'PC{i+1}' for i in range(n_comp)])
            new_data['Attack Type'] = attacks.values

            _save_pickle(scaler, 'scaler', 3)
            _save_pickle(ipca,   'pca',    3)
            self._log("✓ Saved: scaler.pkl, pca.pkl")

            # SMOTE:
            # Digunakan untuk menyeimbangkan jumlah sampel antar kelas.
            # Kelas dipilih hanya dari kelas valid yang menjadi target multi-class.
            # SMOTE — whitelist kelas valid, Unknown tidak bisa masuk
            VALID_CLASSES = {'BENIGN', 'DoS', 'DDoS', 'Port Scan', 'Brute Force', 'Web Attack', 'Bot'}
            cc = new_data['Attack Type'].value_counts()
            sel_classes = [c for c in cc[cc > 1950].index if c in VALID_CLASSES]
            self._log(f"Kelas terpilih ({len(sel_classes)}): {sel_classes}")
            sel = new_data[new_data['Attack Type'].isin(sel_classes)].copy()
            dfs = []
            for name in sel_classes:
                df_c = sel[sel['Attack Type'] == name]
                if len(df_c) > 2500: df_c = df_c.sample(5000, random_state=0)
                dfs.append(df_c)
            df_bal = pd.concat(dfs, ignore_index=True)
            self._prog(65)

            X = df_bal.drop('Attack Type', axis=1)
            y = df_bal['Attack Type']
            X_up, y_up = SMOTE(random_state=0).fit_resample(X, y)
            self._log(f"Setelah SMOTE:\n{pd.Series(y_up).value_counts().to_string()}")
            self._prog(78)

            X_train, X_test, y_train, y_test = train_test_split(X_up, y_up, test_size=0.25, random_state=0)
            self._log(f"Train: {X_train.shape}, Test: {X_test.shape}")

            p3 = os.path.join(OUTPUT_DIR, "phase3")
            np.save(os.path.join(p3, "X_train.npy"), X_train)
            np.save(os.path.join(p3, "X_test.npy"),  X_test)
            np.save(os.path.join(p3, "y_train.npy"), np.array(y_train))
            np.save(os.path.join(p3, "y_test.npy"),  np.array(y_test))

            le = LabelEncoder(); le.fit(y_up)
            _save_pickle(le, 'label_encoder', 3)
            self._prog(90)

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            pd.Series(y_up).value_counts().plot(kind='barh', ax=ax, color='steelblue')
            ax.set_title('Class Distribution After SMOTE')
            _save_plot('smote_class_distribution', 3)

            _save_json({'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'pca_components': n_comp, 'variance_retained_pct': float(var*100),
                        'selected_classes': list(sel_classes),
                        'train_shape': list(X_train.shape), 'test_shape': list(X_test.shape)},
                       'feature_engineering_summary', 3)
            self._log("✓ Phase 3 selesai!")
            self._prog(100)
            self._finish()
        except Exception:
            self._error(f"Phase 3 Error:\n{traceback.format_exc()}")

    # ── Phase 4 ───────────────────────────────────────────────────────────────
    # PHASE 4: MODEL TRAINING
    # Fungsi:
    # 1. Membaca data train-test dari Phase 3.
    # 2. Melatih baseline model: Random Forest, Decision Tree, dan KNN.
    # 3. Melatih proposed model: CNN-LSTM.
    # 4. Menyimpan model baseline sebagai .pkl.
    # 5. Menyimpan model CNN-LSTM sebagai .h5.
    # 6. Menyimpan ringkasan training dan grafik training history.
    def run_phase4(self):
        try:
            self._set_phase(4)
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.model_selection import cross_val_score
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping, Callback
            from tensorflow.keras.utils import to_categorical
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

            class FileOnlyEpochLogger(Callback):
                # Callback ini menyimpan detail epoch CNN-LSTM ke file log saja.
                # Tidak dikirim ke UI agar process log tetap ringkas.
                def __init__(self, file_log_func):
                    super().__init__()
                    self.file_log_func = file_log_func

                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    def fmt(v):
                        return "nan" if v is None else f"{float(v):.4f}"
                    self.file_log_func(
                        "CNN-LSTM Epoch "
                        f"{epoch + 1}: "
                        f"loss={fmt(logs.get('loss'))}, "
                        f"accuracy={fmt(logs.get('accuracy'))}, "
                        f"val_loss={fmt(logs.get('val_loss'))}, "
                        f"val_accuracy={fmt(logs.get('val_accuracy'))}"
                    )

            self._log("=" * 60); self._log("PHASE 4: MODEL TRAINING"); self._log("=" * 60)
            self._prog(2)

            p3 = os.path.join(OUTPUT_DIR, "phase3")
            X_train = np.load(os.path.join(p3, "X_train.npy"))
            X_test  = np.load(os.path.join(p3, "X_test.npy"))
            y_train = np.load(os.path.join(p3, "y_train.npy"), allow_pickle=True)
            y_test  = np.load(os.path.join(p3, "y_test.npy"),  allow_pickle=True)
            le      = _load_pickle('label_encoder', 3)
            self._log(f"Data: X_train {X_train.shape}")
            self._prog(5)

            results = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'run_id': self.run_id,
                'full_log': self._full_log_path,
                'current_log': self._current_log_path,
                'models': {}
            }

            # Random Forest:
            # Baseline ensemble learning berbasis banyak decision tree.
            # Dipakai sebagai pembanding kuat untuk data tabular.
            # RF
            self._log("Training Random Forest...")
            rf1 = RandomForestClassifier(n_estimators=10, max_depth=6, max_features=None, random_state=0)
            rf1.fit(X_train, y_train)
            cv1 = cross_val_score(rf1, X_train, y_train, cv=5)
            rf2 = RandomForestClassifier(n_estimators=10, max_depth=7, max_features=5, random_state=0)
            rf2.fit(X_train, y_train)
            cv2 = cross_val_score(rf2, X_train, y_train, cv=5)
            self._log(f"RF M1 CV: {cv1.mean():.4f} | RF M2 CV: {cv2.mean():.4f}")
            _save_pickle(rf1, 'rf_model1', 4); _save_pickle(rf2, 'rf_model2', 4)
            results['models']['random_forest'] = {
                'model1': {'cv_mean': float(cv1.mean())}, 'model2': {'cv_mean': float(cv2.mean())}
            }
            self._prog(25)

            # Decision Tree:
            # Baseline sederhana dan interpretable.
            # Dipakai untuk melihat performa model pohon tunggal.
            # DT
            self._log("Training Decision Tree...")
            dt1 = DecisionTreeClassifier(max_depth=4)
            dt1.fit(X_train, y_train)
            cv_dt1 = cross_val_score(dt1, X_train, y_train, cv=5)
            dt2 = DecisionTreeClassifier(max_depth=5)
            dt2.fit(X_train, y_train)
            cv_dt2 = cross_val_score(dt2, X_train, y_train, cv=5)
            self._log(f"DT M1 CV: {cv_dt1.mean():.4f} | DT M2 CV: {cv_dt2.mean():.4f}")
            _save_pickle(dt1, 'dt_model1', 4); _save_pickle(dt2, 'dt_model2', 4)
            results['models']['decision_tree'] = {
                'model1': {'cv_mean': float(cv_dt1.mean())}, 'model2': {'cv_mean': float(cv_dt2.mean())}
            }
            self._prog(40)

            # KNN:
            # Baseline berbasis kedekatan jarak antar data.
            # Dipakai sebagai pembanding non-parametric.
            # KNN
            self._log("Training KNN...")
            knn1 = KNeighborsClassifier(n_neighbors=10); knn1.fit(X_train, y_train)
            cv_k1 = cross_val_score(knn1, X_train, y_train, cv=5)
            knn2 = KNeighborsClassifier(n_neighbors=15); knn2.fit(X_train, y_train)
            cv_k2 = cross_val_score(knn2, X_train, y_train, cv=5)
            self._log(f"KNN M1 CV: {cv_k1.mean():.4f} | KNN M2 CV: {cv_k2.mean():.4f}")
            _save_pickle(knn1, 'knn_model1', 4); _save_pickle(knn2, 'knn_model2', 4)
            results['models']['knn'] = {
                'model1': {'cv_mean': float(cv_k1.mean())}, 'model2': {'cv_mean': float(cv_k2.mean())}
            }
            self._prog(55)

            # CNN-LSTM:
            # Proposed deep learning model.
            # Pada versi ini, CNN-LSTM hanya menggunakan SATU model final.
            # Arsitektur yang digunakan adalah konfigurasi terbaik yang sebelumnya
            # disebut sebagai Model 2. Penyederhanaan ini dilakukan agar output
            # Phase 4 dan Phase 5 hanya menampilkan satu hasil CNN-LSTM.
            #
            # Peran layer:
            # - Conv1D: ekstraksi pola lokal dari feature sequence hasil PCA.
            # - MaxPooling1D: reduksi dimensi feature map dan noise.
            # - LSTM: menangkap dependency/sequential representation.
            # - Dense: lapisan klasifikasi sebelum output.
            # - Dropout: mengurangi risiko overfitting.
            # - Softmax: menghasilkan probabilitas untuk setiap kelas serangan.
            self._log("Training CNN-LSTM...")
            y_enc = le.transform(y_train)
            n_cls = len(le.classes_)
            y_cat = to_categorical(y_enc, n_cls)
            X_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

            def make_cnn_lstm():
                # Arsitektur final CNN-LSTM.
                # Konfigurasi ini adalah konfigurasi yang sebelumnya disebut Model 2.
                m = Sequential([
                    Conv1D(160, 5, activation='relu', input_shape=(X_cnn.shape[1], 1)),
                    MaxPooling1D(2),
                    Conv1D(96, 3, activation='relu'),
                    MaxPooling1D(2),
                    LSTM(112, dropout=0.32, recurrent_dropout=0.32),
                    Dense(160, activation='relu'),
                    Dropout(0.42),
                    Dense(n_cls, activation='softmax')
                ])
                m.compile(
                    optimizer=Adam(0.0005),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                return m

            # EarlyStopping dipakai agar training berhenti otomatis jika val_loss
            # tidak membaik. restore_best_weights=True memastikan bobot terbaik
            # yang digunakan adalah bobot pada epoch dengan performa validasi terbaik.
            es = EarlyStopping(
                monitor='val_loss',
                patience=18,
                restore_best_weights=True,
                verbose=0
            )

            cnn_lstm = make_cnn_lstm()

            # Simpan ringkasan arsitektur model ke file log.
            self._write_log_file_only("CNN-LSTM MODEL SUMMARY:")
            cnn_lstm.summary(print_fn=lambda line: self._write_log_file_only(line))

            history = cnn_lstm.fit(
                X_cnn,
                y_cat,
                validation_split=0.2,
                epochs=75,
                batch_size=128,
                callbacks=[es, FileOnlyEpochLogger(self._write_log_file_only)],
                verbose=0
            )

            best_val = float(max(history.history['val_accuracy']))
            self._log(f"CNN-LSTM best val_acc: {best_val:.4f}")

            cnn_model_path = os.path.join(OUTPUT_DIR, "phase4", "models", "cnn_lstm_model.h5")
            cnn_lstm.save(cnn_model_path)
            self._prog(92)

            # Visualisasi training history CNN-LSTM final.
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            axs[0].plot(history.history['accuracy'], label='Train')
            axs[0].plot(history.history['val_accuracy'], label='Val')
            axs[0].set_title('CNN-LSTM: Accuracy'); axs[0].legend()
            axs[1].plot(history.history['loss'], label='Train')
            axs[1].plot(history.history['val_loss'], label='Val')
            axs[1].set_title('CNN-LSTM: Loss'); axs[1].legend()
            plt.tight_layout(); _save_plot('cnn_lstm_training_history', 4)

            results['models']['cnn_lstm'] = {
                'best_val_accuracy': best_val,
                'architecture': {
                    'conv1d_1_filters': 160,
                    'conv1d_1_kernel_size': 5,
                    'conv1d_2_filters': 96,
                    'conv1d_2_kernel_size': 3,
                    'lstm_units': 112,
                    'lstm_dropout': 0.32,
                    'lstm_recurrent_dropout': 0.32,
                    'dense_units': 160,
                    'dropout': 0.42,
                    'optimizer': 'Adam',
                    'learning_rate': 0.0005,
                    'epochs': 75,
                    'batch_size': 128,
                    'validation_split': 0.2,
                    'early_stopping_patience': 18
                },
                'saved_model': 'phase4/models/cnn_lstm_model.h5'
            }
            _save_json(results, 'model_training_summary', 4)
            self._log("✓ Phase 4 selesai!")
            self._prog(100)
            self._finish()
        except Exception:
            self._error(f"Phase 4 Error:\n{traceback.format_exc()}")

    # ── Phase 5 ───────────────────────────────────────────────────────────────
    # PHASE 5: PERFORMANCE EVALUATION
    # Fungsi:
    # 1. Memuat model yang sudah disimpan pada Phase 4.
    # 2. Melakukan prediksi pada test set.
    # 3. Menghitung accuracy, confusion matrix, precision, recall, dan F1-score.
    # 4. Membuat visualisasi perbandingan performa model.
    # 5. Menentukan model terbaik berdasarkan akurasi test.
    def run_phase5(self):
        try:
            self._set_phase(5)
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
            from tensorflow.keras.models import load_model

            self._log("=" * 60); self._log("PHASE 5: EVALUATION"); self._log("=" * 60)
            self._prog(5)

            p3 = os.path.join(OUTPUT_DIR, "phase3")
            X_test  = np.load(os.path.join(p3, "X_test.npy"))
            y_test  = np.load(os.path.join(p3, "y_test.npy"), allow_pickle=True)
            X_train = np.load(os.path.join(p3, "X_train.npy"))
            y_train = np.load(os.path.join(p3, "y_train.npy"), allow_pickle=True)
            le      = _load_pickle('label_encoder', 3)
            self._prog(10)

            rf1  = _load_pickle('rf_model1',  4); rf2  = _load_pickle('rf_model2',  4)
            dt1  = _load_pickle('dt_model1',  4); dt2  = _load_pickle('dt_model2',  4)
            knn1 = _load_pickle('knn_model1', 4); knn2 = _load_pickle('knn_model2', 4)
            self._prog(20)
            cnn_lstm = load_model(os.path.join(OUTPUT_DIR, "phase4", "models", "cnn_lstm_model.h5"))
            self._prog(30)

            X_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            tn    = list(le.classes_)

            y_rf2  = rf2.predict(X_test)
            y_dt2  = dt2.predict(X_test)
            y_knn2 = knn2.predict(X_test)
            y_cnn  = le.inverse_transform(np.argmax(cnn_lstm.predict(X_cnn, verbose=0), axis=1))
            self._prog(55)

            labels = ['Random Forest', 'Decision Tree', 'KNN', 'CNN-LSTM']
            preds  = [y_rf2, y_dt2, y_knn2, y_cnn]
            scores = [accuracy_score(y_test, p) for p in preds]
            for l, s in zip(labels, scores):
                self._log(f"  {l}: {s:.4f}")

            # Accuracy bar
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.barh(labels, scores, color=sns.color_palette('Blues', 4))
            ax.set_xlim([0, 1]); ax.set_xlabel('Test Accuracy')
            ax.set_title('Model Comparison')
            for i, v in enumerate(scores): ax.text(v + 0.002, i, f'{v:.4f}', va='center')
            plt.tight_layout(); _save_plot('final_accuracy_comparison', 5)
            self._prog(65)

            # Confusion matrices
            cms = [confusion_matrix(y_test, p) for p in preds]
            fig, axs = plt.subplots(2, 2, figsize=(24, 20))
            for ax, cm, title in zip(axs.flat, cms, ['Random Forest M2', 'Decision Tree M2', 'KNN M2', 'CNN-LSTM']):
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=tn, yticklabels=tn)
                ax.set_title(title); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
            plt.tight_layout(); _save_plot('final_confusion_matrices', 5)
            self._prog(80)

            # Classification report heatmaps
            datas = []
            for model_label, pred in zip(labels, preds):
                self._write_log_file_only(f"CLASSIFICATION REPORT - {model_label}:")
                self._write_log_file_only(classification_report(y_test, pred, target_names=tn))
                rpt = classification_report(y_test, pred, target_names=tn, output_dict=True)
                datas.append(np.array([[rpt[n]['precision'] for n in tn],
                                       [rpt[n]['recall']    for n in tn],
                                       [rpt[n]['f1-score']  for n in tn]]))
            fig, axs = plt.subplots(2, 2, figsize=(20, 12))
            for ax, d, t in zip(axs.flat, datas, labels):
                sns.heatmap(d, cmap='Pastel1', annot=True, fmt='.2f',
                            xticklabels=tn, yticklabels=['Precision', 'Recall', 'F1'], ax=ax)
                ax.set_title(f'Classification Report: {t}')
            plt.tight_layout(); _save_plot('detailed_classification_reports', 5)
            self._prog(93)

            best_idx = scores.index(max(scores))
            self._log(f"\nBEST MODEL: {labels[best_idx]} (Acc={scores[best_idx]:.4f})")

            # ── FPR & FNR per kelas (revisi Bu Rahma No. 2) ─────────────────────
            # Dihitung dengan pendekatan one-vs-rest dari confusion matrix
            # multiclass yang sudah ada (cms), tanpa perlu prediksi ulang.
            #   FPR = FP / (FP + TN)  -> trafik non-kelas-X yang salah dianggap kelas X
            #   FNR = FN / (FN + TP)  -> trafik kelas-X yang gagal terdeteksi
            self._log("=" * 60); self._log("FPR & FNR PER KELAS (one-vs-rest)"); self._log("=" * 60)

            def _fpr_fnr_per_class(cm, class_names):
                cm = np.asarray(cm, dtype=np.int64)
                total = cm.sum()
                out = {}
                for i, cname in enumerate(class_names):
                    tp = int(cm[i, i])
                    fn = int(cm[i, :].sum() - tp)
                    fp = int(cm[:, i].sum() - tp)
                    tn = int(total - tp - fn - fp)
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                    out[cname] = {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                                  'FPR': float(fpr), 'FNR': float(fnr)}
                return out

            fpr_fnr_all = {}
            for model_label, cm in zip(labels, cms):
                per_class = _fpr_fnr_per_class(cm, tn)
                fpr_fnr_all[model_label] = per_class
                self._write_log_file_only(f"FPR/FNR - {model_label}:")
                for cname, m in per_class.items():
                    self._write_log_file_only(
                        f"  {cname:<12} FPR={m['FPR']:.4f}  FNR={m['FNR']:.4f}"
                    )
                # ringkas macro (rata-rata antar kelas) tetap dikirim ke UI
                macro_fpr = np.mean([m['FPR'] for m in per_class.values()])
                macro_fnr = np.mean([m['FNR'] for m in per_class.values()])
                self._log(f"  {model_label}: macro-FPR={macro_fpr:.4f} | macro-FNR={macro_fnr:.4f}")

            # Visualisasi: grid 2x2 (satu panel per model), masing-masing
            # menampilkan bar FPR vs FNR per kelas.
            fig, axs = plt.subplots(2, 2, figsize=(16, 10))
            x = np.arange(len(tn))
            width = 0.35
            for ax, model_label in zip(axs.flat, labels):
                fpr_vals = [fpr_fnr_all[model_label][c]['FPR'] for c in tn]
                fnr_vals = [fpr_fnr_all[model_label][c]['FNR'] for c in tn]
                ax.bar(x - width/2, fpr_vals, width, label='FPR', color='#e74c3c')
                ax.bar(x + width/2, fnr_vals, width, label='FNR', color='#3498db')
                ax.set_xticks(x); ax.set_xticklabels(tn, rotation=30, ha='right')
                ax.set_ylim([0, max(0.05, max(fpr_vals + fnr_vals) * 1.3)])
                ax.set_title(f'FPR & FNR per Kelas — {model_label}')
                ax.legend()
                for xi, (fv, nv) in enumerate(zip(fpr_vals, fnr_vals)):
                    ax.text(xi - width/2, fv, f'{fv:.3f}', ha='center', va='bottom', fontsize=8)
                    ax.text(xi + width/2, nv, f'{nv:.3f}', ha='center', va='bottom', fontsize=8)
            plt.tight_layout(); _save_plot('fpr_fnr_per_class', 5)
            self._prog(97)

            _save_json({'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'run_id': self.run_id,
                        'full_log': self._full_log_path,
                        'current_log': self._current_log_path,
                        'accuracies': dict(zip(labels, [float(s) for s in scores])),
                        'best_model': {'name': labels[best_idx], 'accuracy': float(scores[best_idx])},
                        'fpr_fnr_per_class': fpr_fnr_all},
                       'evaluation_results', 5)

            # File JSON terpisah khusus FPR/FNR supaya mudah disitasi di laporan
            _save_json({'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'definition': {
                            'FPR': 'FP / (FP + TN) -- proporsi data non-kelas-X yang salah diklasifikasikan sebagai kelas X',
                            'FNR': 'FN / (FN + TP) -- proporsi data kelas-X yang gagal terdeteksi sebagai kelas X'
                        },
                        'per_model_per_class': fpr_fnr_all},
                       'fpr_fnr_metrics', 5)

            p5 = os.path.join(OUTPUT_DIR, "phase5")
            for name, cm in zip(['rf', 'dt', 'knn', 'cnn_lstm'], cms):
                np.save(os.path.join(p5, f"confusion_matrix_{name}.npy"), cm)

            self._log("✓ Phase 5 selesai!")
            self._prog(100)
            self._finish()
        except Exception:
            self._error(f"Phase 5 Error:\n{traceback.format_exc()}")

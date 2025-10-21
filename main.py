import os
import json
import librosa
import numpy as np
import sounddevice as sd
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog
from utils.interface import load_artifacts, predict_file, LABELS

# Rutas (ajústalas si cambiaste estructura)
MODEL_PATH = "/home/leonardo/Documents/GitHub/DCASEv2/models/saved_lstm_model/model.keras"
SCALER_PATH = "/home/leonardo/Documents/GitHub/DCASEv2/artifacts/scaler.joblib"
HISTORY_FILE = "/home/leonardo/Documents/GitHub/DCASEv2/audio_history.json"

class AudioClassifierApp(ttk.Window):
    def __init__(self):
        super().__init__(themename="superhero")
        self.title("Urban Sound Classifier")
        self.geometry("1000x640")
        self.minsize(940, 580)

        self.model = None
        self.scaler = None
        self.current_audio = None
        self.history = self._load_history()

        self._build_ui()
        self._load_artifacts()

    # ----------------- UI -----------------
    def _build_ui(self):
        # Header
        header = ttk.Label(
            self, text="Urban Sound Scene Classifier",
            font=("Segoe UI", 20, "bold"), anchor="center",
            bootstyle="inverse-primary", padding=12
        )
        header.pack(fill=X, pady=(10, 15))

        # Contenedor principal
        main = ttk.Frame(self, padding=18)
        main.pack(fill=BOTH, expand=True)

        # Panel izquierdo: Historial
        left = ttk.Labelframe(main, text="History", padding=10, bootstyle="secondary")
        left.pack(side=LEFT, fill=Y, padx=(0, 16))

        self.history_list = ttk.Treeview(
            left, columns=("name",), show="headings", height=20, bootstyle="dark"
        )
        self.history_list.heading("name", text="Previously loaded files")
        self.history_list.pack(fill=BOTH, expand=True)
        self.history_list.bind("<Double-1>", self._on_history_double_click)
        self._refresh_history_list()

        # Panel derecho: Control + Info
        right = ttk.Frame(main)
        right.pack(side=LEFT, fill=BOTH, expand=True)

        # Barra de acciones
        actions = ttk.Frame(right)
        actions.pack(fill=X, pady=(0, 12))

        ttk.Button(actions, text="Load WAV", bootstyle=PRIMARY, command=self._load_audio)\
            .pack(side=LEFT, padx=6)
        ttk.Button(actions, text="Play", bootstyle=SUCCESS, command=self._play_audio)\
            .pack(side=LEFT, padx=6)
        ttk.Button(actions, text="Predict", bootstyle=INFO, command=self._predict)\
            .pack(side=LEFT, padx=6)
        ttk.Button(actions, text="Stop", bootstyle=WARNING, command=lambda: sd.stop())\
            .pack(side=LEFT, padx=6)

        # Panel de estado/selección actual
        status = ttk.Labelframe(right, text="Current selection", padding=12)
        status.pack(fill=X)

        self.selected_label = ttk.Label(
            status, text="No audio selected",
            font=("Segoe UI", 11), bootstyle="light"
        )
        self.selected_label.pack(fill=X)

        # Resultado
        result_box = ttk.Labelframe(right, text="Prediction", padding=12)
        result_box.pack(fill=BOTH, expand=True, pady=(12, 0))

        self.pred_label = ttk.Label(result_box, text="—", font=("Segoe UI", 16, "bold"))
        self.pred_label.pack(anchor=W, pady=(6, 10))

        self.conf_label = ttk.Label(result_box, text="Confidence: —", font=("Segoe UI", 11))
        self.conf_label.pack(anchor=W)

        # Mensajes / errores
        self.msg_label = ttk.Label(
            result_box, text="", font=("Segoe UI", 10), bootstyle="secondary"
        )
        self.msg_label.pack(anchor=W, pady=(12, 0))

        # Footer
        footer = ttk.Label(
            self, text="Model: LSTM per-frame • 16 kHz mono • MFCC+Contrast+RMS",
            font=("Segoe UI", 9), bootstyle="secondary"
        )
        footer.pack(side=BOTTOM, pady=8)

    # ----------------- Funcionalidad -----------------
    def _load_artifacts(self):
        try:
            self.model, self.scaler = load_artifacts(MODEL_PATH, SCALER_PATH)
            self._set_message("Artifacts loaded.", ok=True)
        except Exception as e:
            self._set_message(f"Error loading artifacts: {e}", ok=False)

    def _load_audio(self):
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if not path:
            return
        self.current_audio = path
        self._set_current_label(path)
        self._add_history(path)

    def _play_audio(self):
        if not self.current_audio:
            self._set_message("Select a WAV file first.", ok=False)
            return
        try:
            y, sr = librosa.load(self.current_audio, sr=16000, mono=True)
            sd.stop()
            sd.play(y, sr)
            self._set_message(f"Playing: {os.path.basename(self.current_audio)}", ok=True)
        except Exception as e:
            self._set_message(f"Playback error: {e}", ok=False)

    def _predict(self):
        if not self.current_audio:
            self._set_message("Select a WAV file first.", ok=False)
            return
        if self.model is None or self.scaler is None:
            self._set_message("Model or scaler not loaded.", ok=False)
            return
        try:
            label, conf = predict_file(self.current_audio, self.model, self.scaler)
            self.pred_label.configure(text=f"Prediction: {label}", bootstyle="success")
            self.conf_label.configure(text=f"Confidence: {conf*100:.2f}%")
            self._set_message("Prediction completed.", ok=True)
        except Exception as e:
            self.pred_label.configure(text="Prediction: —", bootstyle="danger")
            self.conf_label.configure(text="Confidence: —")
            self._set_message(f"Prediction error: {e}", ok=False)

    # ----------------- Historial -----------------
    def _load_history(self):
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_history(self):
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(self.history, f, indent=2)

    def _add_history(self, path):
        if path not in self.history:
            self.history.append(path)
            self._save_history()
            self._refresh_history_list()

    def _refresh_history_list(self):
        for it in self.history_list.get_children():
            self.history_list.delete(it)
        for p in self.history:
            self.history_list.insert("", END, values=(os.path.basename(p),))

    def _on_history_double_click(self, _evt):
        sel = self.history_list.selection()
        if not sel:
            return
        idx = self.history_list.index(sel[0])
        if 0 <= idx < len(self.history):
            self.current_audio = self.history[idx]
            self._set_current_label(self.current_audio)
            self._set_message(f"Selected from history: {os.path.basename(self.current_audio)}", ok=True)

    # ----------------- Helpers UI -----------------
    def _set_current_label(self, path):
        self.selected_label.configure(
            text=f"Selected: {os.path.basename(path)}",
            bootstyle="light"
        )

    def _set_message(self, msg, ok=True):
        style = "success" if ok else "danger"
        self.msg_label.configure(text=msg, bootstyle=style)


if __name__ == "__main__":
    app = AudioClassifierApp()
    app.mainloop()

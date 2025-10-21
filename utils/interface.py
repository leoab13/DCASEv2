import os
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

LABELS = [
    "airport", "shopping_mall", "metro_station", "street_pedestrian",
    "public_square", "street_traffic", "tram", "bus", "metro", "park"
]


def load_artifacts(model_path: str, scaler_path: str):
    """Carga modelo (.keras) y scaler (joblib)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def _features_per_frame(y, sr, frame_length=2048, hop_length=512):
    """Extrae características por frame del audio."""
    features = []
    for i in range(0, len(y), hop_length):
        frame = y[i:i + frame_length]
        if len(frame) < frame_length:
            break

        mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=13)
        spec_contrast = librosa.feature.spectral_contrast(y=frame, sr=sr)
        spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=frame, sr=sr))
        rms = np.mean(librosa.feature.rms(y=frame))

        hist, _ = np.histogram(frame, bins=256)
        prob = hist / np.sum(hist) if np.sum(hist) > 0 else np.zeros_like(hist)
        ent = -np.sum(prob * np.log2(prob + 1e-9))

        feats = np.concatenate(
            (np.mean(mfcc, axis=1), np.mean(spec_contrast, axis=1),
             [spec_bandwidth, rms, ent])
        )
        features.append(feats)

    return np.array(features)


def _pad_or_truncate(seq_2d, target_len):
    """Ajusta la secuencia (T,F) a longitud fija target_len."""
    T, F = seq_2d.shape
    if T == target_len:
        return seq_2d
    if T > target_len:
        return seq_2d[:target_len, :]
    pad = np.zeros((target_len - T, F), dtype=seq_2d.dtype)
    return np.vstack([seq_2d, pad])


def predict_file(audio_path: str, model_or_path, scaler_or_path):
    """
    Predice la escena de un archivo WAV.
    Puede recibir rutas o los objetos ya cargados.
    Retorna (label_str, confidence_float).
    """
    try:
        # Si se pasan rutas, las carga
        if isinstance(model_or_path, str):
            model = load_model(model_or_path)
        else:
            model = model_or_path

        if isinstance(scaler_or_path, str):
            scaler = joblib.load(scaler_or_path)
        else:
            scaler = scaler_or_path

        # Carga del audio
        if not isinstance(audio_path, str) or not os.path.exists(audio_path):
            raise ValueError(f"Ruta inválida: {audio_path}")

        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        features = _features_per_frame(y, sr)
        if features.size == 0:
            return "Error: no se pudieron extraer características", 0.0

        features_scaled = scaler.transform(features)
        features_padded = pad_sequences([features_scaled], dtype="float32",
                                        padding="post", maxlen=400)

        preds = model.predict(features_padded, verbose=0)
        pred_label = np.argmax(preds, axis=1)[0]
        confidence = np.max(preds)

        label_mapping = {i: lbl for i, lbl in enumerate(LABELS)}
        return label_mapping[pred_label], confidence

    except Exception as e:
        return f"Error al predecir: {e}", 0.0

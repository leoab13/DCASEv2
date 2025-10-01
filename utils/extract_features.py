import os
import librosa
import numpy as np
import pandas as pd
from scipy.stats import entropy

# Mapeo de etiquetas a partir del nombre del archivo
label_mapping = {
    "airport": 0,
    "shopping_mall": 1,
    "metro_station": 2,
    "street_pedestrian": 3,
    "public_square": 4,
    "street_traffic": 5,
    "tram": 6,
    "bus": 7,
    "metro": 8,
    "park": 9
}

def extract_features_per_frame(y, sr, hop_length=512, n_mfcc=13):
    """Extrae features por frame"""
    features = {}
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length).T
    
    # Spectral contrast
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length).T
    
    # Spectral bandwidth
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length).T
    
    # RMS
    rms = librosa.feature.rms(y=y, hop_length=hop_length).T
    
    # Entropy: lo calculamos por frame a partir de la energía
    frame_entropy = []
    frame_size = hop_length
    for i in range(0, len(y), frame_size):
        frame = y[i:i+frame_size]
        if len(frame) == 0:
            frame_entropy.append(0)
        else:
            hist, _ = np.histogram(frame, bins=256)
            frame_entropy.append(entropy(hist + 1e-6))  # evitar log(0)
    frame_entropy = np.array(frame_entropy)[:len(mfccs)]  # recortar al mismo número de frames
    
    # Alinear longitudes
    min_len = min(len(mfccs), len(spec_contrast), len(spec_bandwidth), len(rms), len(frame_entropy))
    mfccs = mfccs[:min_len]
    spec_contrast = spec_contrast[:min_len]
    spec_bandwidth = spec_bandwidth[:min_len]
    rms = rms[:min_len]
    frame_entropy = frame_entropy[:min_len]
    
    # Construir matriz final [frames x features]
    features = np.hstack([
        mfccs,
        spec_contrast,
        spec_bandwidth,
        rms,
        frame_entropy.reshape(-1, 1)
    ])
    
    return features

def process_dataset_no_labels(input_dir, output_csv, sr=16000):
    all_data = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                filepath = os.path.join(root, file)
                print(f"Procesando: {file}")

                try:
                    y, _ = librosa.load(filepath, sr=sr, mono=True)
                    features = extract_features_per_frame(y, sr)

                    # Agregar fila por cada frame (sin etiqueta)
                    for frame_idx, frame_feats in enumerate(features):
                        row = [file, frame_idx] + frame_feats.tolist()
                        all_data.append(row)

                except Exception as e:
                    print(f"Error procesando {file}: {e}")

    # Nombres de columnas (sin etiqueta)
    columns = (
        ["audio_id", "frame"] +
        [f"mfcc_{i+1}" for i in range(13)] +
        [f"spec_contrast_{i+1}" for i in range(7)] +
        ["spec_bandwidth", "rms", "entropy"]
    )

    df = pd.DataFrame(all_data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Características guardadas en {output_csv}")



def process_dataset(input_dir, output_csv, sr=16000):
    all_data = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                filepath = os.path.join(root, file)
                print(f"Procesando: {file}")

                try:
                    y, _ = librosa.load(filepath, sr=sr, mono=True)
                    features = extract_features_per_frame(y, sr)

                    # Obtener etiqueta desde el nombre
                    label = None
                    for key in label_mapping.keys():
                        if key in file:
                            label = label_mapping[key]
                            break

                    if label is None:
                        print(f"No se encontró etiqueta para {file}, saltando.")
                        continue

                    # Agregar fila por cada frame
                    for frame_idx, frame_feats in enumerate(features):
                        row = [file, frame_idx] + frame_feats.tolist() + [label]
                        all_data.append(row)

                except Exception as e:
                    print(f"Error procesando {file}: {e}")

    # Definir nombres de columnas
    columns = (
        ["audio_id", "frame"] +
        [f"mfcc_{i+1}" for i in range(13)] +
        [f"spec_contrast_{i+1}" for i in range(7)] +
        ["spec_bandwidth", "rms", "entropy", "label"]
    )

    df = pd.DataFrame(all_data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Características guardadas en {output_csv}")

if __name__ == "__main__":
    os.makedirs("data/features", exist_ok=True)
    #process_dataset("C:\\Users\\edgar\\Documents\\GitHub\\DCASEv2\\data\\processed\\train\\audio", "C:\\Users\\edgar\\Documents\\GitHub\\DCASEv2\\data\\features\\train.csv")
    process_dataset_no_labels("C:\\Users\\edgar\\Documents\\GitHub\\DCASEv2\\data\\processed\\test", "C:\\Users\\edgar\\Documents\\GitHub\\DCASEv2\\data\\features\\test.csv")
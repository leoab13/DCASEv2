# Urban Sound Scene Classification – DCASE 2024 (Project: DCASEv2)
Urban Sound Scene Classification – DCASE 2024 (Project: DCASEv2)

## Project Overview

This project tackles the challenge of classifying urban acoustic scenes (e.g., airport, bus, tram, street traffic) using the dataset from DCASE 2024 Challenge.
By converting raw audio recordings into tabular sequential features and applying a LSTM-based neural network, the model achieves efficient training and inference even under limited computational resources.

Motivation

Large-scale audio datasets and deep-learning models often require extensive compute and time.

- In academic or startup environments, resources may be constrained.
- This project reduces the data footprint by over 99% (from ~30 GB raw audio to compact feature matrices), enabling fast iteration while maintaining strong classification accuracy.
- The solution is designed to be practical, deployable (exported model), and ready for real-time use (desktop GUI + prediction pipeline).

## Key Features

- Preprocessing pipeline: audio conversion (mono, 16 kHz) → frame-based feature extraction (MFCCs, spectral contrast, bandwidth, entropy, RMS)

- Sequential data representation per audio file (frames × features) for LSTM input

- LSTM architecture: 2 layers → Dense → Softmax; exportable model (.keras)

- Desktop application (Tkinter/ttkbootstrap) for interactive use: load audio, playback, predict scene, view confidence

- Persistent history of processed audios, user-friendly UI, modern theme

- Modular codebase: scripts for preprocessing, training, inference and GUI

- Ready for deployment: model artifacts parsed and usable in broader applications (desktop/web/docker)

## Repository Structure

```bash
DCASEv2/
│── data/
│   ├── raw/                     # Original audio data (train/test)
│   ├── processed/               # Converted audio (mono 16 kHz)
│   └── features/                # CSVs of extracted features
│
│── models/
│   ├── saved_lstm_model/        # Exported model (.keras) & scaler
│   ├── train_lstm.py            # Training script
│
│── utils/
│   ├── convert_audio.py         # Audio conversion
│   ├── extract_features.py      # Feature extraction pipeline
│   └── inference_core.py        # Model & scaler loader + prediction logic
│
│── main.py                      # GUI desktop application
│── requirements.txt             # Python dependencies
│── README.md                    # This file

```
## Results & Performance

- Training set reduced from ~30.5 GB to ~64.7 MB via feature extraction and tabular representation (~99.79% reduction)

- Final validation accuracy: ~72.8% on unseen audio scenes

- Metrics available: confusion matrix, classification report, training curves

- Model ready for deployment (exports and inference script included)

NOTE: Full training logs, figure outputs and metrics are stored in results/ folder (see results/confusion_matrix.png, results/train_curves.png, etc.)

## Why This Approach?

- Efficiency: Frame-based tabular features drastically reduce data size and speed up training.

- Temporal modelling: LSTM leverages sequential nature of audio features (frames over time) versus static per-file vectors.

- Deployable: Model and UI built for real-time inference, ideal for prototyping in constrained environments (mobile, edge, research labs).

- Modular: All components (pre-process, train, infer, GUI) are decoupled and reusable in wider contexts (web service, docker container, mobile app).

## How to Run

Clone the repository:
```bash
git clone https://github.com/leoab13/DCASEv2.git
cd DCASEv2
```
Create & activate virtual environment (Python 3.12 recommended):
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Convert audio files:
```bash
python utils/convert_audio.py
```
Extract features:
```bash
python utils/extract_features.py
```
Train model (optional):
```bash
python models/train_lstm.py
```
Launch GUI for inference:
```bash
python main.py
```
## Deployment Note

For production or web deployment, consider containerizing:

- Dockerfile can wrap the GUI or a REST API.

- Model (.keras) is self-contained and ready for usage in backend services.

- Minimal latency due to compact input features → ideal for edge scenarios.

## About the Author

Leonardo A. (GitHub: leoab13)
Data Science / Audio ML / Research Intern
Passionate about efficient machine learning solutions, particularly in audio and embedded systems.
LinkedIn: www.linkedin.com/in/edgar-leonardo-aguirre-bautista-bb7708315

# models/train_lstm.py
import os
import json
import numpy as np
import  pandas as pd
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Masking, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger

# ---------- CONFIG ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "features")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved_lstm_model")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")

RANDOM_SEED = 42
TEST_SIZE = 0.2
BATCH_SIZE = 16
EPOCHS = 200
PATIENCE = 8
# ----------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# -------------- UTIL --------------
def load_train_csv_grouped(csv_path):
    """
    Lee el CSV de train (por frames) y devuelve:
      - sequences: list de arrays [n_frames, n_features]
      - labels: list de int (una por audio)
      - audio_ids: list de audio_id strings (orden correspondiente)
    CSV esperado: audio_id, frame, feat1, feat2, ..., featN, label
    """
    df = pd.read_csv(csv_path)
    # columnas que no son features
    non_feat = {"audio_id", "frame", "label"}
    feat_cols = [c for c in df.columns if c not in non_feat]
    grouped = df.groupby("audio_id")
    sequences = []
    labels = []
    audio_ids = []
    for aid, g in tqdm(grouped, desc="Agrupando audios"):
        arr = g.sort_values("frame")[feat_cols].values.astype(np.float32)
        # obtener label único (asumimos que todas las filas del audio tienen la misma label)
        lab = int(g["label"].iloc[0])
        sequences.append(arr)
        labels.append(lab)
        audio_ids.append(aid)
    return sequences, np.array(labels, dtype=np.int32), audio_ids, feat_cols

# ------------- CARGA Y PREPROCESADO -------------
print("Cargando y agrupando CSV...")
sequences, labels, audio_ids, FEATURE_COLS = load_train_csv_grouped(TRAIN_CSV)
n_features = len(FEATURE_COLS)
n_classes = len(np.unique(labels))
print(f"Audios: {len(sequences)}, features por frame: {n_features}, clases: {n_classes}")

# dividir en train/val a nivel audio
X_train_seq, X_val_seq, y_train, y_val = train_test_split(sequences, labels,
                                                          test_size=TEST_SIZE,
                                                          random_state=RANDOM_SEED,
                                                          stratify=labels)

# Fit scaler sobre todos los frames de X_train
print("Entrenando StandardScaler sobre frames de entrenamiento...")
all_frames = np.vstack([s for s in X_train_seq])  # [total_frames, n_features]
scaler = StandardScaler()
scaler.fit(all_frames)
joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
print("Scaler guardado.")

# Aplicar scaler a cada secuencia
def scale_sequences(sequences, scaler):
    scaled = []
    for s in sequences:
        scaled.append(scaler.transform(s))
    return scaled

X_train_seq = scale_sequences(X_train_seq, scaler)
X_val_seq = scale_sequences(X_val_seq, scaler)

# Padding (usaremos pad_sequences de Keras, padding='post', value=0.0)
print("Padding de secuencias (post)...")
X_train_padded = pad_sequences(X_train_seq, padding="post", dtype="float32", value=0.0)
X_val_padded = pad_sequences(X_val_seq, padding="post", dtype="float32", value=0.0)

print("Shapes: X_train:", X_train_padded.shape, "X_val:", X_val_padded.shape)

# -------------- MODELO --------------
def build_lstm_model(n_features, n_classes):
    inp = Input(shape=(None, n_features), name="input_seq")
    x = Masking(mask_value=0.0)(inp)
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(32, return_sequences=False)(x)
    x = Dense(16, activation="relu")(x)
    out = Dense(n_classes, activation="softmax")(x)
    model = Model(inputs=inp, outputs=out)
    return model

model = build_lstm_model(n_features, n_classes)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Callbacks: CSVLogger+EarlyStopping+ReduceLROnPlateau
csv_log_path = os.path.join(RESULTS_DIR, "training_log.csv")
csv_logger = CSVLogger(csv_log_path)
early = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

# -------------- ENTRENAMIENTO --------------
history = model.fit(
    X_train_padded, y_train,
    validation_data=(X_val_padded, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[csv_logger, early, reduce_lr],
    verbose=1
)

# Guardar modelo en SavedModel (carpeta) y en HDF5 opcional
#model.save(MODEL_DIR)  # SavedModel
model.save(os.path.join(MODEL_DIR, "model.keras"))

# Guardar history (loss/acc) a TXT (formato simple)
log_txt = os.path.join(RESULTS_DIR, "training_log.txt")
with open(log_txt, "w") as f:
    f.write("epoch,loss,accuracy,val_loss,val_accuracy\n")
    for i in range(len(history.history["loss"])):
        f.write(f"{i+1},{history.history['loss'][i]:.6f},{history.history['accuracy'][i]:.6f},"
                f"{history.history['val_loss'][i]:.6f},{history.history['val_accuracy'][i]:.6f}\n")
print("Logs guardados en", log_txt)

# -------------- EVALUACIÓN --------------
print("Evaluando en validación...")
y_pred_probs = model.predict(X_val_padded, batch_size=BATCH_SIZE)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_val, y_pred)
print("Val Accuracy:", acc)

report = classification_report(y_val, y_pred, output_dict=True)
report_txt = os.path.join(RESULTS_DIR, "classification_report.json")
with open(report_txt, "w") as f:
    json.dump(report, f, indent=2)
print("Classification report guardado en", report_txt)

# Matriz de confusión y figura
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (validation)")
plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print("Matriz de confusión guardada en", cm_path)

# Curvas de entrenamiento
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.title("Loss")
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend()
plt.title("Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "train_curves.png"))
plt.close()
print("Curvas de entrenamiento guardadas.")

# Guardar label mapping (si quieres usar indices->label string)
# Aquí asumimos que las etiquetas son 0..K-1 con el mapping definido en extracción.
label_map = {
    0: "airport",
    1: "shopping_mall",
    2: "metro_station",
    3: "street_pedestrian",
    4: "public_square",
    5: "street_traffic",
    6: "tram",
    7: "bus",
    8: "metro",
    9: "park"
}
with open(os.path.join(ARTIFACTS_DIR, "label_map.json"), "w") as f:
    json.dump(label_map, f, indent=2)
print("Label map guardado.")

print("Entrenamiento y evaluación finalizados.")

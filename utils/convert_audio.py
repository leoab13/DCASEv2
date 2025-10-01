import os
import librosa
import soundfile as sf

def convert_to_mono_and_resample(input_file, output_file, sr=16000):
    """Convierte un archivo a mono y 16 kHz"""
    y, _ = librosa.load(input_file, sr=sr, mono=True)
    sf.write(output_file, y, sr, subtype='PCM_16')

def convert_all_audio(input_dir, output_dir, sr=16000):
    """
    Recorre todas las carpetas en input_dir y convierte los .wav
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                in_path = os.path.join(root, file)

                # Estructura de salida mantiene la misma subcarpeta
                rel_path = os.path.relpath(in_path, input_dir)
                out_path = os.path.join(output_dir, rel_path)

                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                print(f"Procesando: {rel_path}")
                convert_to_mono_and_resample(in_path, out_path, sr)
    print("Conversión completada.")

if __name__ == "__main__":
    convert_all_audio("C:\\Users\\edgar\\Documents\\GitHub\\DCASEv2\\data\\raw\\test\\audio", "C:\\Users\\edgar\\Documents\\GitHub\\DCASEv2\\data\\processed\\test")
    convert_all_audio("C:\\Users\\edgar\\Documents\\GitHub\\DCASEv2\\data\\raw\\train\\audio",  "C:\\Users\\edgar\\Documents\\GitHub\\DCASEv2\\data\\processed\\train\\audio")

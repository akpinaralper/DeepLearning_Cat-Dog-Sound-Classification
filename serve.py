import torch
import gradio as gr
import librosa
import numpy as np
from model import AudioCNN
CUSTOM_CSS = """
footer {display: none !important;}
"""


model = AudioCNN()
model.load_state_dict(torch.load("audio_model.pth"))
model.eval()

def predict(audio):
    if audio is None:
        return "Lütfen bir ses dosyası yükleyin."

    file_path = audio

    x, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)
    mfcc = librosa.util.fix_length(mfcc, size=20, axis=1)

    mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        out = model(mfcc)
        pred = out.argmax(1).item()

    return "🐱 Kedi sesi" if pred == 0 else "🐶 Köpek sesi"

interface = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="Ses Sınıflandırma (Kedi/Köpek)",
    flagging_mode="never",
    css=CUSTOM_CSS

)

interface.launch()

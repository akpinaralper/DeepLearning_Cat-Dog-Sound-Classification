import os
import torch
import gradio as gr
import librosa
import numpy as np
from model import AudioCNN

CUSTOM_CSS = "footer {display:none !important;}"


model = AudioCNN()
model.load_state_dict(torch.load("audio_model.pth", map_location="cpu"))
model.eval()

def predict(audio_path):
    if not audio_path:
        return "Lütfen bir ses dosyası seçin."

    x, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)
    mfcc = librosa.util.fix_length(mfcc, size=20, axis=1)
    mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        out = model(mfcc)
        pred = out.argmax(1).item()

    return "🐱 Kedi sesi" if pred == 0 else "🐶 Köpek sesi"


EX_DIR = os.path.join(os.path.dirname(__file__), "examples")
examples = [
    [os.path.join(EX_DIR, "cat_4.wav")],
    [os.path.join(EX_DIR, "cat_68.wav")],
    [os.path.join(EX_DIR, "dog_barking_1.wav")],
    [os.path.join(EX_DIR, "dog_barking_2.wav")],
    [os.path.join(EX_DIR, "dog_barking_3.wav")],
]


with gr.Blocks() as demo:
    gr.Markdown("## Kedi / Köpek Sesi Sınıflandırma")

    audio_in = gr.Audio(type="filepath", label="Ses Dosyası")
    out_box = gr.Textbox(label="Tahmin")

    btn = gr.Button("Tahmin Et")
    btn.click(fn=predict, inputs=audio_in, outputs=out_box)

    gr.Examples(
        examples=examples,
        inputs=audio_in,
        fn=predict,
        outputs=out_box,
        label="Örnek Sesler (tıkla seç)",
        cache_examples=False  
    )

demo.launch(share=True, css=CUSTOM_CSS)

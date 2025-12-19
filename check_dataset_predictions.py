import os
import torch
import librosa
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from model import AudioCNN

DATASET_DIR = os.path.join("dataset", "test")  # TEST seti

model = AudioCNN()
model.load_state_dict(torch.load("audio_model.pth", map_location="cpu"))
model.eval()

def extract_mfcc(path):
    x, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)
    mfcc = librosa.util.fix_length(mfcc, size=20, axis=1)
    mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,40,20)
    return mfcc

y_true = []
y_pred = []

print("\n===== TEST PREDICTIONS =====")

for cls_name, true_label in [("cat", 0), ("dog", 1)]:
    cls_dir = os.path.join(DATASET_DIR, cls_name)
    print(f"\n== TEST/{cls_name.upper()} dosyaları ==")

    for f in sorted(os.listdir(cls_dir)):
        if not f.endswith(".wav"):
            continue

        path = os.path.join(cls_dir, f)
        x = extract_mfcc(path)

        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1)[0].numpy()
            pred = int(probs.argmax())

        y_true.append(true_label)
        y_pred.append(pred)

        label_text = "KEDİ" if pred == 0 else "KÖPEK"
        print(f"{f} -> {label_text}  (p_cat={probs[0]:.2f}, p_dog={probs[1]:.2f})")

# --- METRICS ---
acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)  # [[TN, FP],[FN, TP]] ama burada sınıf sırası: 0(cat),1(dog)

print("\n===== METRICS =====")
print(f"Test Accuracy: {acc*100:.2f}%")

print("\nConfusion Matrix (rows=true, cols=pred) [cat,dog]:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_true, y_pred,
    target_names=["cat", "dog"],
    digits=4
))

# Hatalı örnek sayısı
wrong = sum(1 for t, p in zip(y_true, y_pred) if t != p)
print(f"Wrong predictions: {wrong} / {len(y_true)}")

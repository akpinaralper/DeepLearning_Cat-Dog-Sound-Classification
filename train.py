import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from model import AudioCNN

DATASET_DIR = "dataset"

class AudioDataset(Dataset):
    def __init__(self, root):
        self.files = []
        self.labels = []

        for label, cls in enumerate(["cat", "dog"]):
            cls_dir = os.path.join(root, cls)
            for f in os.listdir(cls_dir):
                if f.endswith(".wav"):
                    self.files.append(os.path.join(cls_dir, f))
                    self.labels.append(label)

    def __getitem__(self, idx):
        file = self.files[idx]
        label = self.labels[idx]

        audio, sr = librosa.load(file, sr=16000)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc = librosa.util.fix_length(mfcc, size=20, axis=1)

        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

        return mfcc, label

    def __len__(self):
        return len(self.files)

def main():
    dataset = AudioDataset(DATASET_DIR)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = AudioCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(15):
        total_loss = 0
        correct = 0

        for mfcc, label in loader:
            optimizer.zero_grad()
            outputs = model(mfcc)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == label).sum().item()

        acc = correct / len(dataset)
        print(f"Epoch {epoch+1} Loss={total_loss:.3f} Acc={acc*100:.1f}%")

    torch.save(model.state_dict(), "audio_model.pth")
    print("Model saved: audio_model.pth")

if __name__ == "__main__":
    main()

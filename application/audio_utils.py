import librosa
import numpy as np
import torch
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image  # Importation de PIL pour manipuler les images
from io import BytesIO  # Importation manquante de BytesIO

import librosa
import numpy as np
import torch
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

def get_mel_spectrogram(filepath):
    # Charger l'audio avec Librosa
    y, sr = librosa.load(filepath)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Supprimer les axes, les échelles et les marges
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.axis('off')  # Supprimer les axes
    fig.tight_layout(pad=0)

    # Sauvegarder l'image du spectrogramme dans un objet BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    mel_spectrogram_img = Image.open(buf)  # Charger l'image depuis le buffer

    # Convertir l'image en 3 canaux (RGB)
    mel_spectrogram_img = mel_spectrogram_img.convert('RGB')

    # Transformation pour l'image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Redimensionner les images pour ResNet18
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisation des images
    ])

    mel_spectrogram = transform(mel_spectrogram_img)
    return mel_spectrogram


def predict_class(model, mel_spectrogram):
    # Ajouter une dimension pour le batch
    mel_spectrogram = mel_spectrogram.unsqueeze(0)

    # Effectuer la prédiction
    with torch.no_grad():
        outputs = model(mel_spectrogram)
        predicted_class = outputs.argmax(1).item()
    return predicted_class

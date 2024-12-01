import streamlit as st
import librosa
import librosa.display
import numpy as np
import torch
import soundfile as sf
import matplotlib.pyplot as plt
from audio_utils import get_mel_spectrogram, predict_class  # Importer les fonctions du fichier audio_utils.py
from dotenv import load_dotenv
import os
import openai
import requests

# Charger les variables d'environnement
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Charger le modèle PyTorch une seule fois avec st.cache_resource
@st.cache_resource(show_spinner=False)
def load_model():
    model = torch.load("application/efficientnetb0_20_epoch.pt")
    model.eval()  # Passer le modèle en mode évaluation
    return model

# Charger le modèle au démarrage
model = load_model()

# Fonction pour appeler l'API Pollinations.ai
def generate_image_from_class(predicted_class):
    # Utiliser la classe prédite comme prompt
    prompt = f"An_image_of_the_number_{predicted_class}_in_an_artistic_style"
    url = f"https://image.pollinations.ai/prompt/{prompt}"

    # Appeler l'API Pollinations.ai
    response = requests.get(url, timeout=60)
    
    if response.status_code == 200:
        st.image(response.content, caption=f"Generated Image for class: {predicted_class}")
        # st.image(response, caption=f"Generated Image for class: {predicted_class}")
    else:
        st.error("Failed to generate image.")
        return None

st.title('Animal Audio to Image and Classification')
st.write('This is a simple web app to convert animal audio to an image and predict its class.')

uploaded_files = st.file_uploader("Choose an audio file", type="wav", accept_multiple_files=True)

if len(uploaded_files) > 0:
    st.write('You selected the following audio file:')
    for uploaded_file in uploaded_files:
        st.audio(uploaded_file, format='audio/wav')

    # Cas où l'utilisateur a chargé un seul fichier audio
    if len(uploaded_files) == 1 and st.button('Convert Audio to Image and Predict Class'):
        # Générer le Mel spectrogram et faire la prédiction
        mel_spectrogram = get_mel_spectrogram(uploaded_files[0])

    
        # Prédire la classe de l'audio
        predicted_class = predict_class(model, mel_spectrogram)

        # Afficher la prédiction
        st.write(f"The predicted class for this audio is: **{predicted_class}**")

        # Générer l'image à partir de la classe prédite
        generate_image_from_class(predicted_class)

    # Cas où l'utilisateur a chargé deux fichiers audio
    elif len(uploaded_files) == 2 and st.button('Mix Audio Files'):
        # Charger les deux fichiers audio
        y1, sr1 = librosa.load(uploaded_files[0])
        y2, sr2 = librosa.load(uploaded_files[1])

        # Vérifier si les taux d'échantillonnage sont identiques
        if sr1 != sr2:
            raise ValueError("Les taux d'échantillonnage des fichiers audio ne correspondent pas.")

        # Ajuster la longueur des deux signaux si nécessaire
        if len(y1) > len(y2):
            y2 = np.pad(y2, (0, len(y1) - len(y2)), 'constant')
        else:
            y1 = np.pad(y1, (0, len(y2) - len(y1)), 'constant')

        # Mélanger les deux signaux
        y_mix = y1 + y2
        sf.write('test/mixed_audio.wav', y_mix, sr1)
        st.audio('test/mixed_audio.wav', format='audio/wav')

        # Générer et afficher le Mel spectrogram du fichier audio mélangé
        mel_spectrogram = get_mel_spectrogram('test/mixed_audio.wav')
        
        # Prédire la classe du fichier audio mélangé
        predicted_class = predict_class(model, mel_spectrogram)

        # Afficher la prédiction
        st.write(f"The predicted class for the mixed audio is: **{predicted_class}**")

        # Générer l'image à partir de la classe prédite
        generate_image_from_class(predicted_class)
        

    # Cas où l'utilisateur a téléchargé plus de deux fichiers
    elif len(uploaded_files) > 2:
        MAX_FILES = 2
        st.warning(f"Maximum number of files reached. Only {MAX_FILES} files can be mixed. Please remove one.")
        uploaded_files = uploaded_files[:MAX_FILES]

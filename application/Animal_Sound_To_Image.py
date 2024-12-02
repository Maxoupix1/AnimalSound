import streamlit as st
import librosa
import librosa.display
import numpy as np
import torch
import soundfile as sf
import matplotlib.pyplot as plt
from audio_utils import get_mel_spectrogram, predict_class  # Importer les fonctions du fichier audio_utils.py
import openai
import config

# Charger les variables d'environnement
openai.api_key = config.api_key_openai()

# Charger le modèle PyTorch une seule fois avec st.cache_resource
@st.cache_resource(show_spinner=False)
def load_model():
    model = torch.load("application/efficientnetb0_20_epoch.pt")
    model.eval()  # Passer le modèle en mode évaluation
    return model

# Charger le modèle au démarrage
model = load_model()

# Fonction pour générer une image via l'API OpenAI
def generate_image_with_openai(prompt):
    try:
        # Appel à l'API OpenAI pour générer une image
        response = openai   .images.generate(
            prompt=prompt,
            n=1,
            size="512x512"  # Taille de l'image
        )
        # Récupérer l'URL de l'image générée
        image_url = response.data[0].url
        print(image_url)
        # Afficher l'image générée dans Streamlit
        st.image(image_url, caption=f"Generated Image for: {prompt}")
    except Exception as e:
        st.error(f"Error generating image: {e}")

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
        prompt = f"An artistic depiction of a {predicted_class} in a creative style"
        generate_image_with_openai(prompt)

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
        prompt = f"An artistic depiction of a {predicted_class} in a creative style"
        generate_image_with_openai(prompt)
        

    # Cas où l'utilisateur a téléchargé plus de deux fichiers
    elif len(uploaded_files) > 2:
        MAX_FILES = 2
        st.warning(f"Maximum number of files reached. Only {MAX_FILES} files can be mixed. Please remove one.")
        uploaded_files = uploaded_files[:MAX_FILES]

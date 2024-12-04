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
def load_model(path):
        model = torch.load(path, map_location=torch.device('cpu'))  # Charger le modèle
        model.eval()  # Passer le modèle en mode évaluation
        return model

# Charger le modèle au démarrage
model_env = load_model("application\\model\\efficeintnet_66_accuracy.pt")
model_animal = load_model("application\\model\\efficientnet_b04.pt")

env_classes = {'Axe': 0,
               'Chainsaw': 1,
               'Clapping': 2,
                'Fire': 3,
                'Firework': 4,
                'Footsteps': 5,
                'Generator': 6,
                'Gunshot': 7,
                'Handsaw' : 8,
                'Helicopter': 9,
                'Rain' : 10,
                'Silence': 11,
                'Speaking human': 12,
                'Thunderstorm': 13,
                'Tree Falling' : 14,
                'Vehicle Engine': 15,
                'Water Drops': 16,
                'Whistling': 17,
                'Wind': 18,
                'Wing Flaping' : 19,
                'Wood Chop' : 20}

animal_classes = {'Bear': 0, 'Cat': 1, 'Chicken': 2, 'Cow': 3, 'Crow': 4, 'Dog': 5, 'Dolphin': 6, 'Donkey': 7, 'Elephant': 8, 'Frog': 9, 'Hen': 10, 'Hiss': 11, 'Horse': 12, 'Tiger': 13, 'Monkey': 14, 'Mouse': 15, 'Pig': 16, 'Rattle': 17, 'Rooster': 18, 'Sheep': 19, 'Whale vocalization': 20}

# Fonction pour générer une image via l'API OpenAI
def generate_image_with_openai(prompt):
    try:
        # Appel à l'API OpenAI pour générer une image
        response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"  # Taille de l'image
        )
        # Récupérer l'URL de l'image générée
        image_url = response.data[0].url
        print(image_url)
        # Afficher l'image générée dans Streamlit
        st.image(image_url)
    except Exception as e:
        st.error(f"Error generating image: {e}")

st.set_page_config(page_title='Animal Audio to Image', page_icon=':lion_face:', layout='centered', initial_sidebar_state='collapsed')
st.title('Animal Audio to Image')
st.logo('application\\images.png', size="large", icon_image="application\\images.png")
st.write('This is a simple web app to convert animal audio to an image and predict its class.')

uploaded_files = st.file_uploader("Choose an audio file", type="wav", accept_multiple_files=True)

if len(uploaded_files) > 0:
    st.write('You selected the following audio file:')
    for uploaded_file in uploaded_files:
        st.audio(uploaded_file, format='audio/wav')

    # Cas où l'utilisateur a chargé un seul fichier audio
    if len(uploaded_files) == 1 :
        st.warning("Please upload another audio file.")
        

    # Cas où l'utilisateur a chargé deux fichiers audio
    elif len(uploaded_files) == 2 and st.button('Mix Audio Files'):
        # Charger les deux fichiers audio
        y1, sr1 = librosa.load(uploaded_files[0])
        y2, sr2 = librosa.load(uploaded_files[1])

        # Vérifier si les taux d'échantillonnage sont identiques
        if sr1 != sr2:
            raise ValueError("Les taux d'échantillonnage des fichiers audio ne correspondent pas.")
        
        # Générer et afficher le Mel spectrogram du fichier audio mélangé
        mel_spectrogram_animal = get_mel_spectrogram(y1, sr1)
        mel_spectrogram_env = get_mel_spectrogram(y2, sr2)

        # Ajuster la longueur des deux signaux si nécessaire
        if len(y1) > len(y2):
            y2 = np.pad(y2, (0, len(y1) - len(y2)), 'constant')
        else:
            y1 = np.pad(y1, (0, len(y2) - len(y1)), 'constant')

        # Mélanger les deux signaux
        y_mix = y1 + y2
        sf.write('test/mixed_audio.wav', y_mix, sr1)
        st.audio('test/mixed_audio.wav', format='audio/wav')
        
        # Prédire la classe du fichier audio mélangé
        
        predicted_class_animal = predict_class(model_animal, mel_spectrogram_animal)
        predicted_class_env = predict_class(model_env, mel_spectrogram_env)

        # Faire l'association entre les classes prédites et les noms de classes
        associated_class_animal = list(animal_classes.keys())[list(animal_classes.values()).index(predicted_class_animal)]
        associated_class_env = list(env_classes.keys())[list(env_classes.values()).index(predicted_class_env)]

        # Afficher la prédiction
        st.write(f"The predicted class for the mixed audio is: **{associated_class_animal}** and **{associated_class_env}**")

        # Générer l'image à partir de la classe prédite
        prompt = f"A depiction of a {associated_class_animal} in a creative but realistic style, in an environment of {associated_class_env} where we can explicitly understand the environment in which the animal is located. Do not put any text on the image."  
        generate_image_with_openai(prompt)
        

    # Cas où l'utilisateur a téléchargé plus de deux fichiers
    elif len(uploaded_files) > 2:
        MAX_FILES = 2
        st.warning(f"Maximum number of files reached. Only {MAX_FILES} files can be mixed. Please remove one.")
        uploaded_files = uploaded_files[:MAX_FILES]

import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


st.title('Animal Audio to Image')
st.write('This is a simple web app to convert animal audio to image')

uploaded_file = st.file_uploader("Choose an audio file", type="wav", accept_multiple_files=False)
if uploaded_file is not None:
    st.write('You selected the following audio file:')
    st.audio(uploaded_file, format='audio/wav')

    st.write('Now you can convert this audio file to image')
    if st.button('Convert to Image'):
        st.write('You clicked the button')
        # Call the function to convert audio to image
    
        # Generate the mel spectrogram 
        y, sr = librosa.load(uploaded_file)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Supprimer les axes, les échelles et les marges
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        ax.axis('off')  # Supprimer les axes
        fig.tight_layout(pad=0)  # Supprimer les marges
        st.pyplot(fig)

        # A FAIRE AVEC LE MODELE
        
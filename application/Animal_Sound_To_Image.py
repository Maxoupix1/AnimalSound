import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


st.title('Animal Audio to Image')
st.write('This is a simple web app to convert animal audio to image')

uploaded_files = st.file_uploader("Choose an audio file", type="wav", accept_multiple_files=True)

if len(uploaded_files) > 0:
    st.write('You selected the following audio file:')
    for uploaded_file in uploaded_files:
            st.audio(uploaded_file, format='audio/wav')
    
    if len(uploaded_files) == 1 and st.button('Convert Audio to Image'):    
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

    if len(uploaded_files) == 2 and st.button('Mix Audio Files'):
        # Mix the audio files  
        y1, sr1 = librosa.load(uploaded_files[0])
        y2, sr2 = librosa.load(uploaded_files[1])

        # Ensure both audio files have the same sample rate
        if sr1 != sr2:
            raise ValueError("Sample rates of the audio files do not match.")
        
        if len(y1) > len(y2):
            y2 = np.pad(y2, (0, len(y1) - len(y2)), 'constant')
        else:
            y1 = np.pad(y1, (0, len(y2) - len(y1)), 'constant')
        
        y_mix = y1 + y2
        sf.write('test/mixed_audio.wav', y_mix, sr1)
        st.audio('test/mixed_audio.wav', format='audio/wav')

        mel_spectrogram = librosa.feature.melspectrogram(y=y_mix, sr=sr1)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Supprimer les axes, les échelles et les marges
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(mel_spectrogram_db, sr=sr1, x_axis='time', y_axis='mel', ax=ax)
        ax.axis('off')
        fig.tight_layout(pad=0)
        st.pyplot(fig)
        

    elif len(uploaded_files) > 2:
        MAX_LINES = 2
        st.warning(f"Maximum number of files reached. Only {MAX_LINES} files can be mixed. Please remove one")
        uploaded_files = uploaded_files[:MAX_LINES]



        
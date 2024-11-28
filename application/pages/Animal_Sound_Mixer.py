import streamlit as st
import librosa
import numpy as np
import soundfile as sf

st.title('Mix multiple audio files')

uploaded_files = st.file_uploader("Choose multiple audio files", type="wav", accept_multiple_files=True)
if uploaded_files is not None:
    st.write('You selected the following audio files:')
    for uploaded_file in uploaded_files:
        st.audio(uploaded_file, format='audio/wav')
    
    st.write('Now you can mix these audio files')
    if st.button('Mix Audio Files'):
        st.write('You clicked the button')
        # Call the function to mix audio files
        
        # Load the audio files
        y1, sr1 = librosa.load(uploaded_files[0])
        y2, sr2 = librosa.load(uploaded_files[1])
        
        # Ensure both audio files have the same sample rate
        if sr1 != sr2:
            raise ValueError("Sample rates of the audio files do not match.")
        
        # Pad the shorter audio file with zeros
        if len(y1) > len(y2):
            y2 = np.pad(y2, (0, len(y1) - len(y2)), 'constant')
        else:
            y1 = np.pad(y1, (0, len(y2) - len(y1)), 'constant')
        
        # Mix the audio files
        y_mix = y1 + y2
        
        # Save the mixed audio file
        sf.write('mixed_audio.wav', y_mix, sr1)
        
        # Display the mixed audio file
        st.audio('mixed_audio.wav', format='audio/wav')
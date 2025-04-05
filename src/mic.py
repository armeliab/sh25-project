import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
from sers import SER, SEA
import os
import torch

#Microphone Input
def record_audio(duration=5):
    """Record audio from microphone"""
    fs = 22050*2  # Sample rate
    recording = st.empty()
    recording.write("Recording...")
    
    # Create progress bar
    progress_bar = st.progress(0)
    for i in range(duration):
        progress_bar.progress((i + 1) / duration)
        if i == 0:  # Only record on first iteration
            audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        st.session_state.recording = True
        st.session_state.audio_data = audio_data
        st.session_state.sample_rate = fs
        sd.sleep(1000)  # Sleep for 1 second
    
    sd.wait()  # Wait for recording to complete
    recording.write("Recording completed!")
    return audio_data, fs

def save_audio(audio_data, sample_rate):
    """Save audio data to a temporary WAV file and return its path"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        sf.write(tmp_file.name, audio_data, sample_rate)
        return tmp_file.name, sample_rate

def main():
    st.title("Simple Voice Recorder")
    st.write("Record and play your voice!")

    ser = SER()
    sea = SEA()

    # Initialize session state
    if 'recording' not in st.session_state:
        st.session_state.recording = False
        st.session_state.audio_data = None
        st.session_state.sample_rate = None
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Recording duration selector
        duration = st.slider("Recording Duration (seconds)", 
                           min_value=3, 
                           max_value=10, 
                           value=5,
                           help="Select how long you want to record")
    
    with col2:
        # Record button
        if st.button("Start Recording"):
            audio_data, sample_rate = record_audio(duration)
            st.session_state.audio_data = audio_data
            st.session_state.sample_rate = sample_rate
    
    # Show audio player if recording exists
    if st.session_state.audio_data is not None:
        st.write("Recorded Audio:")
        # Save audio to temporary file and create audio player
        temp_audio_file, fs = save_audio(st.session_state.audio_data, 
                                   st.session_state.sample_rate)
        st.audio(temp_audio_file)

        '''
        prediction = ser.analyse(temp_audio_file, fs)
        print(type(prediction[0]))
        emotion = max(prediction, key=lambda x: x["score"])
        st.write(emotion["label"])
        '''

        label = sea.analyse(temp_audio_file, fs)
        st.write(label)
        
        # Clean up temporary file
        if os.path.exists(temp_audio_file):
            os.unlink(temp_audio_file)

if __name__ == "__main__":
    main() 

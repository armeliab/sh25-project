import asyncio
import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os
import librosa
import torch
import json
import hume_ai as h
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')
from login import login_page  
from manageFirebase import store_emotion_log, display_flagged_warning 

# Model path (downloaded using download_model.py)
MODEL_PATH = "./model"
FILE_PATH="responses.json"
EMOTION_PATH="emotions.json"
MAX_DURATION = 5  # seconds

class EmotionHealthAdvisor:
    def __init__(self):
        with open(FILE_PATH, "r", encoding="utf-8") as f:
            self.advice_database = json.load(f)  # returns a dict
        
        self.general_wellness_tips = [
            "Regular exercise is good for both physical and mental health.",
            "Getting enough sleep is crucial for emotional regulation.",
            "Meditation or yoga can help improve mental health.",
            "A healthy diet helps maintain emotional stability."
        ]

    def get_advice(self, emotion_results):
        """Provides customized advice based on emotion analysis results."""
        if not emotion_results:
            return {
                'primary_emotion': 'unknown',
                'confidence': 0,
                'specific_advice': ["No emotion analysis results found."],
                'general_tip': "Try recording again."
            }
        
        # Find the emotion with highest probability
        top_emotion = max(emotion_results, key=lambda x: x['score'])
        emotion = top_emotion['name']
        
        # Select advice for the detected emotion
        specific_advice = self.advice_database.get(emotion, ["No specific advice available for this emotion."])
        
        # Choose a random general wellness tip
        general_tip = np.random.choice(self.general_wellness_tips)
        
        return {
            'primary_emotion': emotion,
            'confidence': top_emotion['score'],
            'specific_advice': specific_advice,
            'general_tip': general_tip
        }
    
    def load_emotion_advice(self, file_path="responses.json"):
        """Load the emotion → advice mapping from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            advice_data = json.load(f)  # returns a dict
        self.advice_database = advice_data
        return advice_data



def get_emotion_emoji(emotion):
    """Return emoji based on detected emotion"""
    emotion_emojis = {
        "fearful": "?",
        "calm": "?",
        "neutral": "?",
        "sad": "?",
        "surprised": "?",
        "happy": "?",
        "angry": "?",
        "disgust": "?"
    }
    return emotion_emojis.get(emotion.lower(), "?")

def record_audio(duration=5):
    """Record audio from microphone"""
    fs = 16000
    recording = st.empty()
    recording.write("?? Recording...")
    
    try:
        progress_bar = st.progress(0)
        for i in range(duration):
            progress_bar.progress((i + 1) / duration)
            if i == 0:
                audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            st.session_state.recording = True
            st.session_state.audio_data = audio_data
            st.session_state.sample_rate = fs
            sd.sleep(1000)
        
        sd.wait()
        recording.write("? Recording completed!")
        return audio_data, fs
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
        return None, None

def save_audio(audio_data, sample_rate):
    """Save audio data to a temporary WAV file and return its path"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate)
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving audio: {str(e)}")
        return None

def main():
    st.title("? Voice Emotion Health Advisor")
    st.write("Record your voice and I'll analyze your emotions and provide personalized advice!")
    
    user_id = login_page()
    if not user_id:
        st.info("Please log in to use the app.")
        return
    
    if "user_email" in st.session_state:
        st.markdown(f"**Logged in as:** {st.session_state['user_email']}")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Please run 'python download_model.py' first.")
        st.stop()
    
    # Initialize session state
    if 'recording' not in st.session_state:
        st.session_state.recording = False
        st.session_state.audio_data = None
        st.session_state.sample_rate = None
        st.session_state.emotion = None
        st.session_state.audio_list = [] 
    
    # Initialize emotion recognition and advisor
    try:
        advisor = EmotionHealthAdvisor()
        advisor.load_emotion_advice(FILE_PATH)
        
    except Exception as e:
        st.error("Failed to initialize emotion recognition model.")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        duration = st.slider("Recording Duration (seconds)", 
                           min_value=3, 
                           max_value=15, 
                           value=5,
                           step=1,
                           help="Select how long you want to record")
    
    with col2:
        if st.button("?? Start Recording"):
            results = []
            predictions = []
            total = 0
            audio_data, sample_rate = record_audio(duration)
            if audio_data is not None:
                audio_data = audio_data.flatten()
                max_samples = MAX_DURATION * sample_rate
                audio_list = [audio_data[i:i+max_samples] for i in range(0, len(audio_data), max_samples)]
                
                st.session_state.audio_data = audio_data.flatten()
                st.session_state.sample_rate = sample_rate
                st.session_state.audio_list = audio_list

                if st.session_state.audio_data is not None:
                    st.write("? Recorded Audio:")
                    temp_audio_file = save_audio(st.session_state.audio_data, 
                                            st.session_state.sample_rate)
                
                    with st.spinner("? Analyzing emotions..."):
                        
                        for audio in audio_list:
                            temp_audio_file = save_audio(audio, 
                                            st.session_state.sample_rate)
                            
                            result = asyncio.run(h.analyse(temp_audio_file))
                            if result.prosody.predictions is not None:
                                results.append(result)

                            if not results:
                                st.session_state.emotion = None
                                st.write("No voice detected.")
                            else:
                                with open(EMOTION_PATH, "r", encoding="utf-8") as f:
                                    st.session_state.emotion = json.load(f)  # returns a dict
                                
                                for result in results:
                                    predictions.append(h.get_emotion(result))
                                for pred in predictions:
                                    for key, score in pred:
                                        dic = next(item for item in st.session_state.emotion if item['name'] == str(key[1]))
                                        dic['score'] += score[1]
                                        total += score[1]

                        
    
    # Show audio player and results if recording exists
    if st.session_state.audio_data is not None:
        st.write("? Recorded Audio:")
        temp_audio_file = save_audio(st.session_state.audio_data, 
                                   st.session_state.sample_rate)
        if temp_audio_file:
            st.audio(temp_audio_file)
            
            if st.session_state.emotion:
                # Display emotion analysis results
                st.write("### ? Emotion Analysis Results")
                
                cols = st.columns(len(st.session_state.emotion[:3]))
                sorted_predictions = sorted(st.session_state.emotion, 
                                         key=lambda x: x['score'], 
                                         reverse=True)
                count = 0
                for i, pred in enumerate(sorted_predictions[:3]):
                    count += 1
                    with cols[i]:
                        emoji = get_emotion_emoji(pred['name'])
                        st.write(f"{emoji} {pred['name'].capitalize()}")
                        st.progress(pred['score']/len(st.session_state.audio_list))
                        st.write(f"{(pred['score']*100)/len(st.session_state.audio_list):.1f}%")
                        
                    if count > 3:
                        break
                
                # Display personalized advice
                st.write("### ? Personalized Advice")
                advice = advisor.get_advice(st.session_state.emotion)
                
                # Display primary emotion and advice
                st.info(f"Primary emotion detected: {get_emotion_emoji(advice['primary_emotion'])} "
                       f"{advice['primary_emotion'].capitalize()} "
                       f"({advice['confidence']*100/len(st.session_state.audio_list):.1f}%)")
                
                # Display specific advice
                st.write("#### ? Advice for You")
                for tip in advice['specific_advice']:
                    st.write(f"? {tip}")
                
                # Display general wellness tip
                st.write("#### ? General Wellness Tip")
                st.write(f"? {advice['general_tip']}")
            
                # Log emotion to Firebase (new addition)
                transcript = st.text_area("Describe how you're feeling:")
                if st.button("Submit Log"):
                    store_emotion_log(user_id, advice['primary_emotion'], advice['specific_advice'][0], transcript)
                    display_flagged_warning(user_id, st)
                    st.success("Emotion log submitted to Firebase.")
                    
            # Clean up temporary file
            try:
                if os.path.exists(temp_audio_file):
                    os.unlink(temp_audio_file)
            except Exception:
                pass

if __name__ == "__main__":
    main() 
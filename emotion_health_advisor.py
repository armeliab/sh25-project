import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os
import librosa
import torch
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Model path (downloaded using download_model.py)
MODEL_PATH = "./model"

class EmotionHealthAdvisor:
    def __init__(self):
        self.advice_database = {
            "fearful": [
                "Fear is a natural emotion. Try to release anxiety through deep breathing.",
                "Remind yourself that you're in a safe place and look around slowly.",
                "When anxious, try having a warm cup of tea to calm yourself.",
                "Writing down your worries in a notebook can help organize your thoughts."
            ],
            "calm": [
                "You're in a great peaceful state. Keep practicing mindfulness.",
                "This is a perfect time to focus on your goals and aspirations.",
                "Use this calm energy to engage in creative activities.",
                "Share your peaceful state with others around you."
            ],
            "neutral": [
                "Maintain your balanced state while practicing mindfulness.",
                "Appreciate this moment and think about things you're grateful for.",
                "How about trying a new hobby or learning something new?",
                "Maintaining a balanced emotional state is very important. You're doing well!"
            ],
            "sad": [
                "Don't stay alone when you're feeling down. Talk to friends or family.",
                "How about listening to your favorite music to comfort yourself?",
                "Try taking a short walk or light exercise to change your mood.",
                "Remember that what you're feeling is temporary. This too shall pass."
            ],
            "surprised": [
                "Take a moment to process your surprise and breathe deeply.",
                "Channel this energy into something productive.",
                "Use this heightened awareness to explore new perspectives.",
                "Share your experience with others if you feel comfortable."
            ],
            "happy": [
                "Recording happy moments in a diary or photo can create good memories for later.",
                "How about sharing this joy with people around you?",
                "Use this positive energy to start something you've been wanting to try.",
                "When you're happy, it's a great time to focus on creative activities!"
            ],
            "angry": [
                "Take deep breaths and exhale slowly. This helps calm your anger.",
                "How about taking a walk to calm your mind?",
                "When angry, creative activities like exercise or drawing can help.",
                "Your emotions are natural, but it's important to express them constructively."
            ],
            "disgust": [
                "Try to identify what's causing this feeling and address it directly.",
                "Focus on pleasant surroundings or memories to shift your perspective.",
                "Consider talking to someone about what's bothering you.",
                "Remember that this feeling will pass with time."
            ]
        }
        
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
        emotion = top_emotion['label'].lower()
        
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

# Original SER class from emotion_mic.py
class SER:
    def __init__(self):
        try:
            self.pipe = pipeline(
                task="audio-classification",
                model=MODEL_PATH,
                device=-1  # Use CPU
            )
            self.sr = 16000  # Fixed sampling rate
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            raise e

    def analyse(self, audio_data, sr):
        temp_file = None
        try:
            if sr != self.sr:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sr)
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(temp_file.name, audio_data, self.sr)
            predictions = self.pipe(temp_file.name)
            return predictions
        except Exception as e:
            st.error(f"Error analyzing audio: {str(e)}")
            return None
        finally:
            if temp_file:
                try:
                    temp_file.close()
                    if os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
                except Exception as e:
                    st.warning(f"Could not cleanup temporary file: {str(e)}")

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
    
    # Initialize emotion recognition and advisor
    try:
        ser = SER()
        advisor = EmotionHealthAdvisor()
    except Exception as e:
        st.error("Failed to initialize emotion recognition model.")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        duration = st.slider("Recording Duration (seconds)", 
                           min_value=3, 
                           max_value=10, 
                           value=5,
                           help="Select how long you want to record")
    
    with col2:
        if st.button("?? Start Recording"):
            audio_data, sample_rate = record_audio(duration)
            if audio_data is not None:
                st.session_state.audio_data = audio_data.flatten()
                st.session_state.sample_rate = sample_rate
                
                with st.spinner("? Analyzing emotions..."):
                    predictions = ser.analyse(st.session_state.audio_data, sample_rate)
                    if predictions:
                        st.session_state.emotion = predictions
                        # Debug output
                        st.write("Debug - Raw predictions:")
                        st.write(predictions)
    
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
                
                cols = st.columns(len(st.session_state.emotion))
                sorted_predictions = sorted(st.session_state.emotion, 
                                         key=lambda x: x['score'], 
                                         reverse=True)
                
                for i, pred in enumerate(sorted_predictions):
                    with cols[i]:
                        emoji = get_emotion_emoji(pred['label'])
                        st.write(f"{emoji} {pred['label'].capitalize()}")
                        st.progress(pred['score'])
                        st.write(f"{pred['score']*100:.1f}%")
                
                # Display personalized advice
                st.write("### ? Personalized Advice")
                advice = advisor.get_advice(st.session_state.emotion)
                
                # Display primary emotion and advice
                st.info(f"Primary emotion detected: {get_emotion_emoji(advice['primary_emotion'])} "
                       f"{advice['primary_emotion'].capitalize()} "
                       f"({advice['confidence']*100:.1f}%)")
                
                # Display specific advice
                st.write("#### ? Advice for You")
                for tip in advice['specific_advice']:
                    st.write(f"? {tip}")
                
                # Display general wellness tip
                st.write("#### ? General Wellness Tip")
                st.write(f"? {advice['general_tip']}")
            
            # Clean up temporary file
            try:
                if os.path.exists(temp_audio_file):
                    os.unlink(temp_audio_file)
            except Exception:
                pass

if __name__ == "__main__":
    main() 
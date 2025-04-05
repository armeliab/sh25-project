import streamlit as st
import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os
import librosa
import torch
from transformers import pipeline
from scipy.signal import butter, filtfilt
import time
import warnings
warnings.filterwarnings('ignore')

# actual code that will be used
# Model path (downloaded using download_model.py)
MODEL_PATH = "./model"

def enhance_audio(audio_data, sample_rate):
    """Enhance audio quality by applying noise reduction and filters"""
    try:
        # 1. Normalize audio
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        # 2. Apply high-pass filter to remove low frequency noise
        nyquist = sample_rate / 2
        cutoff = 80  # Hz
        order = 4
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        audio_data = filtfilt(b, a, audio_data)
        
        return audio_data
        
    except Exception as e:
        st.error(f"Error enhancing audio: {str(e)}")
        return audio_data

def save_audio_wav(audio_data, sample_rate):
    """Save audio data to a WAV file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            audio_data = audio_data.astype(np.float32)
            audio_data = audio_data / np.max(np.abs(audio_data))
            sf.write(tmp_file.name, audio_data, sample_rate)
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving audio: {str(e)}")
        return None

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

def main():
    st.title("? Enhanced Voice Emotion Analyzer")
    st.write("Record your voice for emotion analysis and speech recognition with enhanced audio quality!")
    
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
        st.session_state.transcribed_text = None
    
    # Initialize emotion recognition and advisor
    try:
        ser = SER()
        advisor = EmotionHealthAdvisor()
    except Exception as e:
        st.error("Failed to initialize emotion recognition model.")
        st.stop()

    # Language selection
    languages = {
        'Korean': 'ko-KR',
        'English': 'en-US',
        'Japanese': 'ja-JP',
        'Chinese': 'zh-CN',
        'Spanish': 'es-ES',
        'French': 'fr-FR',
        'German': 'de-DE'
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_language = st.selectbox(
            "Select Language",
            options=list(languages.keys()),
            index=0
        )
    
    with col2:
        duration = st.slider("Recording Duration (seconds)", 
                           min_value=3, 
                           max_value=10, 
                           value=5,
                           help="Select how long you want to record")
    
    with col3:
        if st.button("?? Start Recording"):
            # Create Recognizer object for speech-to-text
            recognizer = sr.Recognizer()

            with sr.Microphone() as source:
                # Adjust for ambient noise and collect noise profile
                with st.spinner("Analyzing background noise..."):
                    recognizer.adjust_for_ambient_noise(source, duration=2)
                
                st.info("Recording will start in 3 seconds. Please speak in a quiet environment.")
                
                # 3-second countdown
                countdown = st.empty()
                for i in range(3, 0, -1):
                    countdown.text(f"Starting in: {i} seconds")
                    time.sleep(1)
                countdown.text("Please speak now!")

                try:
                    # Record audio
                    audio = recognizer.listen(source, timeout=duration)
                    st.info("Processing your voice...")

                    # Save to temporary WAV file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                        temp_audio.write(audio.get_wav_data())
                        temp_audio_path = temp_audio.name

                    # Read WAV file
                    audio_data, sample_rate = sf.read(temp_audio_path)

                    # Noise reduction and audio enhancement
                    enhanced_audio = enhance_audio(audio_data, sample_rate)
                    
                    # Save enhanced audio
                    enhanced_audio_path = save_audio_wav(enhanced_audio, sample_rate)

                    if enhanced_audio_path:
                        # Store audio data in session state
                        st.session_state.audio_data = enhanced_audio
                        st.session_state.sample_rate = sample_rate

                        # Display audio playback
                        st.write("? Recorded Audio:")
                        st.audio(enhanced_audio_path)

                        # Convert enhanced audio to text
                        with sr.AudioFile(enhanced_audio_path) as source:
                            enhanced_audio_data = recognizer.record(source)
                            text = recognizer.recognize_google(
                                enhanced_audio_data, 
                                language=languages[selected_language]
                            )
                            st.session_state.transcribed_text = text

                        # Perform emotion analysis
                        with st.spinner("? Analyzing emotions..."):
                            predictions = ser.analyse(enhanced_audio, sample_rate)
                            if predictions:
                                st.session_state.emotion = predictions

                        # Clean up temporary files
                        try:
                            os.unlink(temp_audio_path)
                            os.unlink(enhanced_audio_path)
                        except Exception:
                            pass

                except sr.WaitTimeoutError:
                    st.error("Timeout: No voice input detected.")
                except sr.UnknownValueError:
                    st.error("Could not recognize the voice. Please try again.")
                except sr.RequestError as e:
                    st.error(f"API Request Error: {e}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    # Show results if we have both transcribed text and emotion analysis
    if st.session_state.transcribed_text and st.session_state.emotion:
        # Display transcribed text
        st.write("### ? Transcribed Text:")
        text_area = st.text_area("", st.session_state.transcribed_text, height=100)
        
        if st.button("? Copy Text"):
            st.code(st.session_state.transcribed_text)
            st.info("Text has been copied!")

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

    # Usage instructions
    with st.expander("?? How to Use"):
        st.markdown("""
        1. Select your language for speech recognition.
        2. Choose recording duration (3-10 seconds).
        3. Click 'Start Recording' and wait for the countdown.
        4. Speak clearly after the prompt.
        5. View your transcribed text and emotion analysis results.
        6. Get personalized advice based on your emotional state.
        
        **Enhanced Features:**
        - Background noise reduction
        - Voice signal optimization
        - Multi-language support
        - Emotion analysis with advice
        - Speech-to-text conversion
        """)

if __name__ == "__main__":
    main() 
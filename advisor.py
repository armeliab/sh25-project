import asyncio
import json
from pyneuphonic import Neuphonic, TTSConfig
from pyneuphonic.player import AudioPlayer
# from google import genai
# from google.genai import types

import streamlit as st

import speech_recognition as sr
import sounddevice as sd
import soundfile as sf

import numpy as np
import hume_rest as h
import tempfile
import os
from scipy.signal import butter, filtfilt

import time
import warnings
warnings.filterwarnings('ignore')

# actual code that will be used
# Model path (downloaded using download_model.py)
MODEL_PATH = "./model"
FILE_PATH="responses.json"
EMOTION_PATH="emotions.json"
MAX_DURATION = 5  # seconds


# Access the variables
key = 'db7231b553380b6b520836e937b1413c0e91640d1d3655e84527d37479f3cfa2.e6808a68-7c07-40ba-931a-5c1bd14a7421'

# Load the API key from the environment
client = Neuphonic(api_key=key)

sse = client.tts.SSEClient()

# TTSConfig is a pydantic model so check out the source code for all valid options
tts_config = TTSConfig(
    speed=1.05,
    lang_code='en', # replace the lang_code with the desired language code.
    voice_id='e564ba7e-aa8d-46a2-96a8-8dffedade48f'  # use client.voices.list() to view all available voices
)

# client = genai.Client(api_key="WRITE API KEY HERE")

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
        emotion = top_emotion['name'].lower()
        
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
    
    '''
    def get_response(self, emotion_results, text_input):
        # Find the emotion with highest probability
        top_emotion = max(emotion_results, key=lambda x: x['score'])
        emotion = top_emotion['name'].lower()

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction="respond with mental health booster advice in a caring tone, make points on how to feel better but not too long for each point"
                                    "maximum three points""flowy sentence between each points"
                                    f"consider the state of the user emotion, that is {emotion}"
            ),
            contents=text_input,
        )
        responses = response.text.strip().replace("*", "")
        return responses
    '''



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
        st.session_state.recorded = False
    
    # Initialize emotion recognition and advisor
    try:
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
                           max_value=15, 
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
                        max_samples = MAX_DURATION * sample_rate
                        # Store audio data in session state
                        st.session_state.audio_data = enhanced_audio
                        st.session_state.sample_rate = sample_rate

                        # Display audio playback
                        st.write("? Recorded Audio:")
                        st.audio(enhanced_audio_path)

                        '''
                        # Convert enhanced audio to text
                        with sr.AudioFile(enhanced_audio_path) as source:
                            enhanced_audio_data = recognizer.record(source)
                            text = recognizer.recognize_google(
                                enhanced_audio_data, 
                                language=languages[selected_language]
                            )
                            st.session_state.transcribed_text = text
                        '''
                        
                        # Perform emotion analysis
                        with st.spinner("? Analyzing emotions..."):
                            if enhanced_audio is not None:
                                results = []
                                max_samples = MAX_DURATION * sample_rate
                                audio_list = []
                                for i in range(1+len(enhanced_audio)//max_samples):
                                    audio_list.append(enhanced_audio[max_samples*i:max_samples*(i+1)])

                                
                                        
                                for audio in audio_list:
                                    temp_audio_file = save_audio_wav(audio, 
                                                    st.session_state.sample_rate)
                                    
                                    result = asyncio.run(h.analyse(temp_audio_file))
                                    try:
                                        result = result.results.predictions[0].models.prosody.grouped_predictions[0].predictions[0].emotions
                                        results.append(result)
                                    except:
                                        pass
                                        

                                if not results:
                                    st.session_state.emotion = None
                                else:
                                    with open(EMOTION_PATH, "r", encoding="utf-8") as f:
                                        st.session_state.emotion = json.load(f)  # returns a dict
                                        
                                    for result in results:
                                        for name, score in result:
                                            dic = next(item for item in st.session_state.emotion if item['name'] == str(name[1]))
                                            dic['score'] += score[1]
                                st.session_state.recorded = True


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
    # if st.session_state.transcribed_text and st.session_state.emotion:
    if st.session_state.recorded and not st.session_state.emotion:
        st.write('## No voice detected. Please record again.')
    if st.session_state.emotion:
        # Display transcribed text
        st.write("### ? Transcribed Text:")
        text_area = st.text_area("", st.session_state.transcribed_text, height=100)
        
        if st.button("? Copy Text"):
            st.code(st.session_state.transcribed_text)
            st.info("Text has been copied!")

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
                st.progress(pred['score']/len(audio_list))
                st.write(f"{pred['score']*100/len(audio_list):.1f}%")
                if count > 3:
                        break
        
        # Display personalized advice
        st.write("### ? Personalized Advice")
        advice = advisor.get_advice(st.session_state.emotion)
        ai_response = advisor.get_response(st.session_state.emotion, st.session_state.transcribed_text)
        
        # Display primary emotion and advice
        st.info(f"Primary emotion detected: {get_emotion_emoji(advice['primary_emotion'])} "
               f"{advice['primary_emotion'].capitalize()} "
               f"({advice['confidence']*100:.1f}%)")
        
        # Display specific advice
        st.write("#### ? Advice for You")
        # for tip in advice['specific_advice']:
        #     st.write(f"? {tip}")
        st.write(ai_response)

        # Create an audio player with `pyaudio`
        with AudioPlayer() as player:
            # response = sse.send(ai_response, tts_config=tts_config)
            response = sse.send(advice, tts_config=tts_config)
            player.play(response)

            player.save_audio('output_audi.wav')  # save the audio to a .wav file
        
        # # Display general wellness tip
        # st.write("#### ? General Wellness Tip")
        # st.write(f"? {advice['general_tip']}")

        

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
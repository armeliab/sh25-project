# Moodify - Voice Emotion Analysis & AI Advisor 🎭

## Inspiration

In a world full of digital interactions, our emotional needs are often overlooked. We wanted to create a system that listens not just to what you say, but how you say it — and responds with empathy, intelligence, and a human touch.

Inspired by the challenge posed by *Neuphonic*, we imagined a future where voice interfaces do more than just respond — they understand, care, and support emotional wellbeing in real time.

## What it does

- 🎙 Records your voice 
- 🧠 Detects your emotional tone using a fine-tuned *Hugging Face SER model*
- 💬 Converts speech to text 
- 🧘‍♂ Uses *Gemini 2.0* to generate personalized, caring advice
- 🔊 Delivers that advice back to the user using *Neuphonic's hyper-realistic Voice AI*
- 🌐 Runs on a clean and responsive *Streamlit UI*

## How we built it

| Component | Technology |
|----------|------------|
| Speech-to-Text | SpeechRecognition |
| Emotion Recognition | Hugging Face Wav2Vec2 (SER model) |
| AI Advice Generation | Gemini 2.0 Flash (Google GenAI) |
| Voice Output | Neuphonic API + TTSConfig |
| Frontend | Streamlit |
| Audio Processing | Butterworth Filter, Noise Reduction (SciPy) |
| (Optional) Database | Firebase Firestore for storing logs |


## GDG Challenge
Total number of team: 4

Project name: Moodify

Google technologies used: Firebase, Google Cloud Firestore api, Gemini API

Problem that could exist in the year 2080:
   In 2080, even with super advanced technologies like AI and flying cars being human doesn't change. We still feel sadness, stress, anxiety, loneliness, fear just like we do today.
   Maybe by 2080, people are overwhelmed by information, virtual realities, aliens(?), or pressure to be constantly connected. No matter how futuristic the world gets, our inner world still needs attention, support, and understanding
   
Proposed solution:
   Our project is to recognize how someone feels and that could be the first step toward healing. By generating a response especially one that’s kind, empathetic, or uplifting it gives people something most tech doesn’t: emotional validation

Code summary:
The code records the user's voice via the microphone, enhances the audio by applying noise filtering, and saves the audio as a file. The recorded voice is then converted into text using speech recognition.
Next, a classification model from Hugging Face is used to predict the user's emotion based on the voice. Based on both the content of the speech and the predicted emotion, personalized mental health advice is generated using Gemini AI. Finally, the generated advice is converted back into audio and played to the user.


## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/moodify.git
cd moodify
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download the emotion recognition model:
```bash
python download_model.py
```

### Usage

1. Start the Streamlit app:
```bash
streamlit run advisor.py
```

2. The app will open in your default web browser.

3. Select your preferred language from the dropdown menu.

4. Choose the recording duration (3-10 seconds).

5. Click "Start Recording" and speak after the countdown.

6. View your results:
   - Transcribed text
   - Emotion analysis
   - Personalized advice

## Challenges we ran into

- 🎧 *Audio processing quirks*: Handling microphone input and ensuring clean audio in noisy environments was tough. We solved this with custom filters and normalization.
- 💬 *Integrating multiple APIs*: Connecting Hugging Face, Gemini, Neuphonic, and Firebase together inside one real-time pipeline took careful coordination.
- 🌍 *Multi-language support*: Mapping audio models with the correct language recognition required trial and error.

## Accomplishments that we're proud of

- Built a functional *Voice AI system* from scratch in under 36 hours
- Seamlessly integrated *emotion detection, advice generation, and voice response*
- Created a calming, responsive, and actually useful experience
- Designed a platform that's both technically robust and emotionally impactful 💛

## What we learned

- Integrating multiple APIs under tight constraints
- How important UX is when dealing with emotionally sensitive topics
- That AI, when designed with empathy, can genuinely comfort people

## What's next for Moodify

- 🔐 *User login* via Firebase (OAuth) to support personalization  
- 🔮 *Emotion trend prediction* using LSTM based on user history  
- 📈 *Dashboard analytics* for tracking mood patterns over time  
- 🧬 *Few-shot personalized coaching* using past emotional context  
- 🗣 *Customizable voice personalities* via Neuphonic agents  
- 🧘‍♀ *Assistant modes* like Calm, Focus, and Listener

## Requirements

The project requires the following main dependencies:

```
streamlit>=1.28.0
speech_recognition>=3.10.0
sounddevice>=0.4.6
soundfile>=0.12.1
numpy>=1.24.3
librosa>=0.10.1
torch>=2.0.1
transformers>=4.35.2
scipy>=1.11.3
```

For a complete list of dependencies, see `requirements.txt`.



## Contributors

- Jiho Park
- Armelia Kasenda
- Jihyun Moon
- Minjun Kim 


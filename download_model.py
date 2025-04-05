from transformers import pipeline
import os

def download_model():
    """Download the emotion recognition model"""
    print("Downloading emotion recognition model...")
    try:
        # Create model directory if it doesn't exist
        if not os.path.exists("./model"):
            os.makedirs("./model")
        
        # Initialize pipeline to download the model
        pipeline(
            task="audio-classification",
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            device=-1
        )
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading model: {e}")

if __name__ == "__main__":
    download_model() 
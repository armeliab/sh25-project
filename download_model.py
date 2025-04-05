from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import os

def download_model():
    """Download the emotion recognition model"""
    print("Downloading emotion recognition model...")
    try:
        # Create model directory if it doesn't exist
        if not os.path.exists("./model"):
            os.makedirs("./model")
        
        # Download model and feature extractor
        model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        
        # Download and save the model
        model = AutoModelForAudioClassification.from_pretrained(model_name)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        
        # Save the model and feature extractor
        model.save_pretrained("./model")
        feature_extractor.save_pretrained("./model")
        
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading model: {e}")

if __name__ == "__main__":
    download_model() 
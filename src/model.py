
import librosa
import torch
from transformers import pipeline

default = "../data/Audio_Speech_Actors_01-24/Actor_01/03-01-01-01-01-01-01.wav"
sampling_rate=16000
device = 0 if torch.cuda.is_available() else -1

class SER:
    def __init__(self):
        self.pipe=pipeline(
                "audio-classification", 
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                device=device)

    def analyse(self, path=default, sr=sampling_rate):
        audio, sr = librosa.load(path)

        predictions = self.pipe(audio)
        return predictions



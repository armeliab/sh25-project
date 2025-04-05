
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
        self.sr=sampling_rate 

    def analyse(self, path=default):
        audio, sr = librosa.load(path, sr=self.sr)

        predictions = self.pipe(audio, sr=sr)
        return predictions

ser = SER()
print(ser.analyse())


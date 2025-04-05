
import librosa
import numpy as np
import torch
from transformers import pipeline
from models import SEA_model

default = "../data/Audio_Speech_Actors_01-24/Actor_01/03-01-01-01-01-01-01.wav"
sampling_rate=50000
device = 0 if torch.cuda.is_available() else -1


class SER:
    def __init__(self):
        self.pipe=pipeline(
                "audio-classification", 
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                device=device)

    def analyse(self, path=default, sr=sampling_rate):
        audio, sr = librosa.load(path, sr=sr)

        predictions = self.pipe(audio)
        return predictions
    
    
class SEA:
    def __init__(self):
        self.model = SEA_model()
        self.model.load_model()
        self.labels = [
            "female_angry",   # index 0
            "female_calm",    # index 1
            "female_fearful", # index 2
            "female_happy",   # index 3
            "female_sad",     # index 4
            "male_angry",     # index 5
            "male_calm",      # index 6
            "male_fearful",   # index 7
            "male_happy",     # index 8
            "male_sad"        # index 9
        ]

    def mfcc(self, path=default, sr=sampling_rate, target_length=216):
        audio, sr = librosa.load(path, sr=sr)
        mfcc = np.mean(librosa.feature.mfcc(y=audio, 
                                                sr=sr, 
                                                n_mfcc=13),
                        axis=0)
        if mfcc.shape[0] < target_length:
            # Zero-pad on the right if it's too short
            pad_width = target_length - mfcc.shape[0]
            mfcc = np.pad(mfcc, (0, pad_width), mode='constant')
        elif mfcc.shape[0] > target_length:
            # Trim if it's too long
            mfcc = mfcc[:target_length]

        # 4. Reshape to (1, target_length, 1) for a single sample
        mfcc = mfcc.reshape(1, target_length, 1)
        return mfcc
    
    def pred(self, mfcc):
        pred = self.model.model.predict(mfcc)
        class_idx = np.argmax(pred[0])
        print(pred)
        predicted_label = self.labels[class_idx]
        return predicted_label
    
    def analyse(self, path=default, sr=sampling_rate, target_length=216):
        m = self.mfcc(path, sr, target_length)
        label = self.pred(m)
        return label
 

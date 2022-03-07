import librosa 
import sounddevice as sd
import wavio
import numpy as np
from model import model


def read_audio(file):
    with open(file, "rb") as audio_file:
        audio_bytes = audio_file.read()
    return audio_bytes


def record(duration=5, fs=48000):
    sd.default.samplerate = fs
    sd.default.channels = 1
    myrecording = sd.rec(int(duration * fs))
    sd.wait(duration)
    return myrecording


def save_record(path_myrecording, myrecording, fs):
    wavio.write(path_myrecording, myrecording, fs, sampwidth=2)
    return None


def extractor(file):
    audio, sample_rate = librosa.load(file, res_type= "kaiser_fast")
    mfccs_transformed = librosa.feature.mfcc(y=audio, sr = sample_rate, n_mfcc=50)
    mfccs_scaled = np.mean(mfccs_transformed.T, axis=0)
    return mfccs_scaled


label = {0: 'dog_bark',
            1: 'children_playing',
            2: 'car_horn',
            3: 'air_conditioner',
            4: 'street_music',
            5: 'gun_shot',
            6: 'siren',
            7: 'engine_idling',
            8: 'jackhammer',
            9: 'drilling',
            10: 'Background'}

def labeling(path):
    recording = extractor(path).reshape(1,-1)
    model_pred = model.predict(recording)
    prediction = np.argmax(model_pred, axis=1)
    for elem in prediction:
        prediction_label = label[elem]
    return prediction_label
    
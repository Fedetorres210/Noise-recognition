import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import IPython.display as ipd 
import pandas as pd
import numpy as np


noise_path = "audio_files/UrbanSound8K/audio/"
background_path = "audio_files/backsounds/"
noise_dataset = pd.read_csv("audio_files/UrbanSound8K/metadata/UrbanSound8K.csv")


# Creation of audio function 
def extractor(file):
    audio, sample_rate = librosa.load(file, res_type= "kaiser_fast")
    mfccs_transformed = librosa.feature.mfcc(y=audio, sr = sample_rate, n_mfcc=50)
    mfccs_scaled = np.mean(mfccs_transformed.T, axis=0)
    return mfccs_scaled


# Creation of dataset and loader

def feature_creator(path_1= noise_path,path_2=background_path,df=noise_dataset):
    features = []
    for index_num,row in df.iterrows():
        file_name = os.path.join(os.path.abspath(path_1),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
        final_class_labels=row["class"]
        data= extractor(file_name)
        features.append([data,final_class_labels])

    for elem in os.listdir(path_2):
        file = os.path.join(os.path.abspath(path_2),elem)
        mccfs = extractor(file)
        features.append([mccfs, "Background"])
    print(f"{len(features)} sounds loaded")
    return features













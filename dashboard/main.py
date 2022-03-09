
import streamlit as st
import numpy as np
import sounddevice as sd
from audio import record, save_record, labeling, read_audio

st.title("Noise Recognition project")


#filename = st.text_input("Chosse a name for your file")
st.text(sd.query_devices())
stop = st.button(f"Click to stop recording")
if st.button(f"Click to record"):
    record_state = st.text("Recording...")
    myrecording = record(10,44100)
    path_myrecording = f"./samples/sample"
    save_record(path_myrecording, myrecording,44100)
    record_state = st.text("Making the predicton ")
    prediction = labeling(path_myrecording)
    st.text(f"This was a {prediction}")
        
        
    st.audio(read_audio(path_myrecording))

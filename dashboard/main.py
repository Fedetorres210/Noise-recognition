import streamlit as st
import numpy as np
from test import model
from audio import record, save_record, read_audio, extractor

st.title("Noise Recognition project")


#filename = st.text_input("Chosse a name for your file")
stop = st.button(f"Click to stop recording")
count = 0
if st.button(f"Click to record"):
    while not stop:
    
        record_state = st.text("Recording...")
        duration = 10
        fs = 44100
        myrecording = record(duration,fs)
        path_myrecording = f"./samples/sample"
        save_record(path_myrecording, myrecording,fs)
        record_state = st.text("Making the predicton ")
        recording = extractor(path_myrecording)
        recording = recording.reshape(1,-1)
        recording = model.predict(recording)
            
        prediction = np.argmax(recording, axis=1)
        record_state = st.text(f"Esta sonando{prediction}")
        
        

        #record_state.text((f"Saving sample as dashboard({count}).wav"))
        
        
        #st.audio(read_audio(path_myrecording))

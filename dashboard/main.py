
from fileinput import filename
from random import sample
import streamlit as st
import numpy as np
import sounddevice as sd
from audio import record, save_record, labeling, read_audio

page = st.sidebar.selectbox('Select-Page',
  ['Information','Prerecorded-predictions','Live-Predictions(In Progress..)'])

if page == 'Information':
    st.title("Noise Recognition project")
    st.text("The idea of this project is to create a model")
    st.text("that is capable of recognizing the different sounds in an environment.")
    st.subheader("Quick explanation")

    st.image(r"dashboard/images/diagram.png")




    st.subheader("Contact Information")
    st.markdown("**Creator: Federico Torres Lobo** ")
    st.image('https://media-exp1.licdn.com/dms/image/C4D03AQGlbB9Mjl6lFw/profile-displayphoto-shrink_200_200/0/1622270211832?e=1652313600&v=beta&t=BE4zuDfi-_HLCfH0LVOwCSz28JQiB8xT0ePQkC6V728')
    st.markdown("[Github](https://github.com/Fedetorres210/Noise-recognition)  [Linkedin](https://www.linkedin.com/in/federico-torres-lobo-729494211/)")
   

      
    






elif page == 'Prerecorded-predictions':
   st.title("Predicting recorded files")
   st.markdown("For this section you can choose the file you want from you computer and uploaded on the 'Drag and drop section' for testing the model.")
   #st.image("https://www.vocesnuestras.org/sites/default/files/styles/800x480/public/img/principal/micro_vn.jpg?itok=6FfLXZKQ")
   audio = st.sidebar.file_uploader("Upload your recording")
   st.sidebar.markdown("Listen your selected file")
   st.sidebar.audio(audio)
   button = st.button("Make the prediction")
   
   if button:
        prediction = labeling(audio)
        st.subheader(f"This was a {prediction}")
    


else:
    st.title("Real Time noise predictions")
    st.markdown("**For this section you can use your microphone to get real time predictions. This function on the web is still on development**")
    st.image('https://images.immediate.co.uk/production/volatile/sites/4/2018/07/iStock_000018226211_Large-6d7599f.jpg?quality=90&resize=940%2C400')
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("**Recording real time**")
        stop = st.button(f"Click to stop recording")
        start = st.button(f"Click to record")
        while start:
                record_state = st.text("Recording...")
                myrecording = record(10,44100)
                path_myrecording = f"./samples/sample"
                save_record(path_myrecording, myrecording,44100)
                record_state = st.text("Making the predicton ")
                prediction = labeling(path_myrecording)
                st.text(f"This was a {prediction}")
                st.audio(read_audio(path_myrecording))
    with col2:
        st.markdown("**Recording one sample and testing**")
        start1 = st.button(f"Click to record one sample")
        if start1:
                record_state = st.text("Recording...")
                myrecording = record(10,44100)
                path_myrecording = f"./samples/sample"
                save_record(path_myrecording, myrecording,44100)
                record_state = st.text("Making the predicton ")
                prediction = labeling(path_myrecording)
                st.text(f"This was a {prediction}")
                sample = read_audio(path_myrecording)
                st.audio(sample)
                st.download_button('Download your audio',sample, "Sample")












#filename = st.text_input("Chosse a name for your file")
    

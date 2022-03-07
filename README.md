[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Visual Studio](https://badgen.net/badge/icon/visualstudio?icon=visualstudio&label)](https://visualstudio.microsoft.com)
[![GitHub](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com)
[![Docker](https://badgen.net/badge/icon/docker?icon=docker&label)](https://https://docker.com/)



# Noise Recognition: Audio Events
Project about a model and a Dashboard focused on noise recognition.
![img](https://www.caracteristicas.co/wp-content/uploads/2016/04/sonido-e1558136756154.jpg)
## Table of Content
---
- [Description](#Description)
- [Installation](#Installation)
    - [Steps](#Steps)
    - [Server run](#Server)
- [How to use it](#How-to-use-it)
- [Tecnologies](#Tecnologies)
    - [Datasets](#Datasets)
    - [Important Libraries](#Important-Libraries)
- [Similar Projects](#Similar-projects)
- [Documentation and Contact](#Documentation-and-Contact)
---
  


## Description
---

 The idea of ​​this project is to create a model that is capable of recognizing the different sounds in an environment. In order to find this model, different models will be used, looking for the one that works best, at the same time the TensorFlow library will be used to create a neural network and compare if the model is better than said network. On the other hand it will be use a dashboard for deploying the model and neural network.

 In this document we will see the progress and future ideas of the project, this also means that the project is still in development

----

## Installation
---
###  Steps



    $git start

    $git clone https://github.com/Fedetorres210/Noise-recognition.git

    $ pip install -r requirements.txt
It's necesary to have installed all the libraries of the requirements.txt file. Once have installed all, the server can be started.

---
### Server 
To run the dashboard, in the same level where you have the file [main.py](dashboard/main.py) , the next command:

    streamlit run main.py

---

## How to use it 
 In progress...

---

## Tecnologies

### Datasets
---
#### **UrbanSound8K Dataset**

This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes: air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music. The classes are drawn from the urban sound taxonomy. For a detailed description enter the following links:   
* <https://urbansounddataset.weebly.com/urbansound8k.html>
* <http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_urbansound_acmmm14.pdf>
---

#### **Isolated Urban Sound**

The Isolated urban sound database contains the audio samples used to design urban sound mixtures using SimScene software.
https://zenodo.org/record/1213793#.YiYcAHrMKUl

--- 
#### **CitySounds2017train audio files**

The CitySounds2017train dataset comprising 1100 1-minute .wav audio files recorded at 44 green infrastructure sites within Greater London, UK between 2013 and 2015.

###  Important Libraries
---
#### **Librosa**
![](https://librosa.org/doc/latest/_static/librosa_logo_text.svg)

[librosa](https://librosa.org/doc/latest/index.html) is a python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems.

---
#### **TensorFlow**
![](https://www.gstatic.com/devrel-devsite/prod/v2484c9574f819dcf3d7ffae39fb3001f4498b2ece38cec22517931d550e19e7d/tensorflow/images/lockup.svg)
[TensorFlow](https://www.tensorflow.org/api_docs/python/tf) is an end-to-end open source platform for machine learning

---
#### **Keras**

![](https://keras.io/img/logo.png)

[Keras](https://www.tensorflow.org/guide/keras?hl=es-419) is TensorFlow's high-level API for building and training deep learning models. It is used for rapid prototyping, cutting-edge (state-of-the-art) research, and in production.


---
#### **SoundDevice**


[SoundDevice](https://python-sounddevice.readthedocs.io/) is a  Python module that  provides bindings for the PortAudio library and a few convenience functions to play and record NumPy arrays containing audio signals.


---
#### **Streamlit**
![](https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png)
[Streamlit](https://docs.streamlit.io/) is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science.

#### **Numpy**
![](https://numpy.org/doc/stable/_static/numpylogo.svg)

[Numpy](https://numpy.org/doc/stable/) is the fundamental package for scientific computing with Python

---
## Similar projects 
---
### Urban street noise recogniton 
* <http://sedici.unlp.edu.ar/bitstream/handle/10915/23897/Documento_completo.pdf?sequence=1>      

   





## Documentation and Contact
- McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. “librosa: Audio and music signal analysis in python.” In Proceedings of the 14th python in science conference, pp. 18-25. 2015.
- Justin Salomon -  <jpbello@nyu.edu>
- Fairbrass, Alison (2018): CitySounds2017train audio files. figshare. Dataset. https://doi.org/10.6084/m9.figshare.5886532.v1 
- Rodríguez, Yohanna; Ballesteros L, Dora Maria; Renza, Diego (2019), “Fake voice recordings (Imitation)”, Mendeley Data, V1, doi: 10.17632/ytkv9w92t6.1
- ![forthebadge](https://zenodo.org/badge/6309729.svg)


FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN apt-get update && apt-get upgrade -y -o Dpkg::Options::="--force-confold"
RUN apt-get install -y apt-utils

RUN apt-get install -y libportaudio2
RUN apt-get autoremove

RUN pip install -r requirements.txt
RUN pip install sounddevice
RUN apt-get install -y libsndfile1
RUN pip install SoundFile

EXPOSE 8501

COPY ./ /app/

ENTRYPOINT ["streamlit", "run"]

CMD ["main.py"]


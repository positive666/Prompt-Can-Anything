import argparse
import functools
import os
import yaml
import numpy as np
import ffmpeg

from pydub import AudioSegment
from pydub.silence import split_on_silence
import soundfile
from utils.audio2face_streaming_utils import push_audio_track_stream,push_audio_track,push_stream
from numba import jit
import pyaudio
import wave
import keyboard
import time
import whisper 
import requests
from ChatGPT.GPT import Chatbot
from ChatGPT.config.private import API_KEY,PROXIES



def speech_recognition(speech_file, model):
    # whisper
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(speech_file)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    speech_language = max(probs, key=probs.get)

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # print the recognized text
    speech_text = result.text
    return speech_text, speech_language



Avatar_instance_A='/World/audio2face/PlayerStreaming'
a2f_url = 'localhost:50051' # The audio2face url by default
sample_rate_Omniverse = 22050 # Audio frame rate
# 录音参数
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS =5
audio_file = "F:\\VoiceprintRecognition-Pytorch-develop\\error001.wav"
buffer_length=int(RATE / CHUNK * RECORD_SECONDS)
record_file='record.wav'
p = pyaudio.PyAudio()  

def mic_audio():
     stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    frames_per_buffer=CHUNK,
                    input=True)
     print("Recording...")
     frames = []
     while True:
          data = stream.read(CHUNK)
          frames.append(data)
          if keyboard.is_pressed('s'):
               break
     stream.stop_stream()
     stream.close()
     p.terminate()

     wf = wave.open(record_file, 'wb')
     wf.setnchannels(CHANNELS)
     wf.setsampwidth(p.get_sample_size(FORMAT))
     wf.setframerate(RATE)
     wf.writeframes(b''.join(frames))
     wf.close()
     #return frames

def send_stream(whisper_modelm,chat_bot):
     
     while True:
          if keyboard.is_pressed('q'):
               mic_audio()
               print('recording done')
               # load speech
               
               print('asr model load done')
               speech_text, speech_language= speech_recognition(record_file,whisper_model)
               for data in chatbot.ask_stream(prompt=speech_text):
                    print(data,end='',flush=True)
          
     #audio_data, samplerate = soundfile.read(audio_file, dtype="float32")
     # if len(audio_data.shape) > 1:
     #   audio_data = np.average(audio_data, axis=1)
     # push_audio_track_stream(a2f_url, audio_data, RATE , Avatar_instance_A) 

if __name__ == "__main__":
      whisper_model = whisper.load_model("base")
      chatbot=Chatbot(api_key=API_KEY,proxy=PROXIES,engine="gpt-3.5-turbo")
      send_stream(whisper_model,chatbot)
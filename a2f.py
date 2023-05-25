import argparse
import functools
import os
import yaml
import numpy as np
import ffmpeg

from pydub import AudioSegment
from pydub.silence import split_on_silence
import soundfile
from audio2face_streaming_utils import push_audio_track_stream,push_audio_track,push_stream
from numba import jit
import pyaudio
import wave 
import time
import whisper 
import requests
from ChatGPT.GPT import Chatbot
from ChatGPT.config.private import API_KEY,PROXIES
chatbot=Chatbot(api_key=API_KEY,proxy=PROXIES,engine="gpt-3.5-turbo")

for data in chatbot.ask_stream(prompt="Hello world"):
     print(data,end='',flush=True)

def check_caption(caption, pred_phrases, max_tokens=100, model="gpt-3.5-turbo",chatbot=None):
    object_list = [obj.split('(')[0] for obj in pred_phrases]
    object_num = []
    for obj in set(object_list):
        object_num.append(f'{object_list.count(obj)} {obj}')
    object_num = ', '.join(object_num)
    print(f"Correct object number: {object_num}")

    prompt = [
        {
            'role': 'system',
            'content': 'Revise the number in the caption if it is wrong. ' + \
                       f'Caption: {caption}. ' + \
                       f'True object number: {object_num}. ' + \
                       'Only give the revised caption: '
        }
    ]
    for data in chatbot.ask_stream(prompt=prompt):
          print(data,end='',flush=True)

          reply = data['choices'][0]['message']['content']
          caption = reply.split(':')[-1].strip()
          return caption




Avatar_instance_A='/World/audio2face/PlayerStreaming'
a2f_url = 'localhost:50051' # The audio2face url by default
sample_rate_Omniverse = 22050 # Audio frame rate
# 录音参数
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS =5
audio_file = "F:\\VoiceprintRecognition-Pytorch-develop\\error001.wav"
buffer_length=int(RATE / CHUNK * RECORD_SECONDS)

p = pyaudio.PyAudio()  


     
def mic_audio():
     stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    frames_per_buffer=CHUNK,
                    input=True)

     frames = []

     for i in range(0, int(RATE  / CHUNK * RECORD_SECONDS)):
          data = stream.read(CHUNK)
          frames.append(data)

     stream.stop_stream()
     stream.close()

     p.terminate()

     wf = wave.open(audio_file, 'wb')
     wf.setnchannels(CHANNELS)
     wf.setsampwidth(p.get_sample_size(FORMAT))
     wf.setframerate(RATE)
     wf.writeframes(b''.join(frames))
     wf.close()
     
def send_stream():
     
     # 调用百度语音合成API进行中文TTS
     text=check_caption("你好！你是谁", pred_phrases, max_tokens=100, model="gpt-3.5-turbo",chatbot=None)
     # 保存TTS结果到文件
     # with open('output.mp3', 'wb') as f:
     #       f.write(audio_data)
     #audio_data, samplerate = soundfile.read(audio_file, dtype="float32")
     if len(audio_data.shape) > 1:
       audio_data = np.average(audio_data, axis=1)
     push_audio_track_stream(a2f_url, audio_data, RATE , Avatar_instance_A) 

if __name__ == "__main__":
      send_stream()
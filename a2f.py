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
from llm_cards.bridge_chatgpt import predict
from config_private import API_KEY
import uuid



# 按秒截取音频
def get_part_wav(sound, start_time, end_time, part_wav_path):
    save_path = os.path.dirname(part_wav_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    start_time = int(start_time) * 1000
    end_time = int(end_time) * 1000
    word = sound[start_time:end_time]
    word.export(part_wav_path, format="wav")


def crop_wav(path, crop_len):
    for src_wav_path in os.listdir(path):
        wave_path = os.path.join(path, src_wav_path)
        print(wave_path[-4:])
        if wave_path[-4:] != '.wav':
            continue
        file = wave.open(wave_path)
        # 帧总数
        a = file.getparams().nframes
        # 采样频率
        f = file.getparams().framerate
        # 获取音频时间长度
        t = int(a / f)
        print('总时长为 %d s' % t)
        # 读取语音
        sound = AudioSegment.from_wav(wave_path)
        for start_time in range(0, t, crop_len):
            save_path = os.path.join(path, os.path.basename(wave_path)[:-4], str(uuid.uuid1()) + '.wav')
            get_part_wav(sound, start_time, start_time + crop_len, save_path)


def speech_recognition(inputs, model,stream_model=False):
    # whisper
    all_result=''
    if not stream_model:
          audio,sr= soundfile.read(inputs, dtype='float32')
    else:  
          print('numpy')
          sr,audio=inputs
    chunk_size=sr*30
    for i in range(0, len(audio), chunk_size):                 
        chunk_end = min(i + chunk_size, len(audio))
        chunk = whisper.pad_or_trim(audio[i:chunk_end])
    # load audio and pad/trim it to fit 30 seconds
   # audio = whisper.load_audio(speech_file)
    
       # chunk= whisper.pad_or_trim(chunk)

          # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(chunk).to(model.device)

          # detect the spoken language
        _, probs = model.detect_language(mel)
        speech_language = max(probs, key=probs.get)

          # decode the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)

    # print the recognized text
        print(result.text)
        all_result+=result.text
    return all_result, speech_language




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

def send_stream(whisper_modelm):
     
     while True:
          if keyboard.is_pressed('q'):
               mic_audio()
               print('recording done')
               # load speech
               
               print('asr model load done')
               speech_text, speech_language= speech_recognition('voice/temp.mp3',whisper_model)
                
          
     #audio_data, samplerate = soundfile.read(audio_file, dtype="float32")
     # if len(audio_data.shape) > 1:
     #   audio_data = np.average(audio_data, axis=1)
     # push_audio_track_stream(a2f_url, audio_data, RATE , Avatar_instance_A) 

if __name__ == "__main__":
      whisper_model = whisper.load_model("small",download_root="weights")
      speech_text, speech_language= speech_recognition('voice_dir/temp.wav',whisper_model)
      print(speech_text)
      print(type(speech_text))
     # chatbot=Chatbot(api_key=API_KEY,proxy=PROXIES,engine="gpt-3.5-turbo")
      #send_stream(whisper_model)
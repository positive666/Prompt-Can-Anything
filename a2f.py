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
import pyaudio
import wave

import time
import whisper 
import requests
from llm_cards.bridge_chatgpt import predict
from config_private import API_KEY
import uuid
import re 
import asyncio


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

from concurrent.futures import ThreadPoolExecutor
def process_chunk(model, chunk, detect_language):
    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(chunk).to(model.device)

    # detect the spoken language
    speech_language = 'zh'
    if detect_language :
        _, probs = model.detect_language(mel)
        speech_language = max(probs, key=probs.get)

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    return result.text, speech_language

def speech_recognition(inputs, model,stream_model=False,detect_language=False):
    # whisper
    all_result=''
    speech_language='zh'
    executor = ThreadPoolExecutor()
    results = []
    audio=None
    if not stream_model:
          audio,sr= soundfile.read(inputs, dtype='float32')
    else:  
          print('numpy  data')
          sr,audio=inputs
          data = audio / 65538
          audio = data.astype(np.float32)
    print(sr)
    chunk_size= sr*30
    print((audio))
    for i in range(0, len(audio), chunk_size):
        chunk_end = min(i + chunk_size, len(audio))
        chunk = whisper.pad_or_trim(audio[i:chunk_end])
        
        # submit the chunk to the thread pool for processing
        results.append(executor.submit(process_chunk, model, chunk, detect_language))

    # print the recognized text and the detected language
    for result in results:
        text, language = result.result()
        #print(text)
        all_result += text
        speech_language = language

    # # print the recognized text
    # all_result+=result.text
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

def mic_audio(record_file="record.wav"):
     # 打开录音
     import keyboard
     stream = p.open(
                input_device_index=1,
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
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
     return 'OK'



import edge_tts
import threading
async def tts_send(text,onmiverse=False,send_file='voice_dir/send_a2f.wav'):
        if text is not None:
            sentences = re.split(r'[！？。: ]', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            sentences_len=len(sentences)
            audio_chunks = {}
            async def process_sentences():
                tasks = []
                for i, sentence in enumerate(sentences):
                    if len(sentence) > 0:
                    # 提交任务到协程池
                        task = asyncio.create_task(speak(sentence, i % sentences_len))
                        tasks.append(task) 
                await asyncio.gather(*tasks) 
                       
            async def speak(sentence, worker_id):
                    # 合成语音
                print(worker_id)
         
                audio_stream =edge_tts.Communicate(sentence, voice='zh-CN-YunxiNeural', rate='+1%', volume='+1%').stream()
                async for package in audio_stream:
                    if package['type'] == 'audio':
                        # 获取音频数据的字节流(chunk)
                        audio_chunk = package['data']
                        # 将音频数据添加到字典中
                        if worker_id not in audio_chunks:
                            audio_chunks[worker_id] = []
                        audio_chunks[worker_id].append(audio_chunk)  
                            
            await process_sentences()
            # 将每个协程合成的音频数据拼接起来
            
            audio_data = b''
            for i in range(sentences_len):
                if i in audio_chunks:
                     for chunk in audio_chunks[i]:
                         audio_data += chunk
            with open(f'{send_file}', 'wb') as f:
                  f.write(audio_data)  
            if onmiverse:
                audio_data, samplerate = soundfile.read(f'{send_file}', dtype="float32")                
                if len(audio_data.shape) > 1:
                    audio_data = np.average(audio_data, axis=1)
                push_audio_track_stream(a2f_url, audio_data, samplerate, Avatar_instance_A)  
                
                
                
import queue
async def tts_a2f(text):
    import edge_tts
    import soundfile as sf
    #import numpy as np
    from  audio2face_streaming_utils import push_audio_track_stream
    generate_wave = edge_tts.Communicate(text, voice='zh-CN-YunxiNeural', rate='-5%', volume='+1%')
    await generate_wave.save('./voice_dir/send_frame.wav')  
                                       
    try:
        audio_data, samplerate = sf.read('./voice_dir/send_frame.wav', dtype="float32")
        if len(audio_data.shape) > 1:
            audio_data = np.average(audio_data, axis=1)
        print("send a2f app....")    
        push_audio_track_stream(a2f_url, audio_data, samplerate , Avatar_instance_A) 

        return "SEND DONE"
    except Exception as e:
        print(f"检查是否开启omniverse!!!")              
                
async def tts_send2(text,onmiverse=False, send_file='voice_dir/send_a2f.wav'):
    # 处理句子列表
    sentences = re.split(r'[！？。,]', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # 将所有句子合并成一段长文本，并合成语音
    whole_text = "".join(sentences)
    audio_stream = edge_tts.Communicate(whole_text, voice='zh-CN-YunxiNeural', rate='+1%', volume='+1%').stream()
    
    # 将音频数据添加到队列中
    audio_queue = queue.Queue()
    async for package in audio_stream:
        if package['type'] == 'audio':
            audio_chunk = package['data']
            audio_queue.put(audio_chunk)

    # 将队列中的音频数据拼接成一段完整的音频
    audio_data = b''
    while not audio_queue.empty():
        audio_data += audio_queue.get()
    
    # 将合成结果保存为 WAV 文件或者发送到服务器
    with open(f'{send_file}', 'wb') as f:
        f.write(audio_data)
    if onmiverse:
        audio_data, samplerate = soundfile.read(f'{send_file}', dtype="float32")                
        if len(audio_data.shape) > 1:
            audio_data = np.average(audio_data, axis=1)
        push_audio_track_stream(a2f_url, audio_data, samplerate, Avatar_instance_A)


import grpc
import audio2face_pb2
import audio2face_pb2_grpc
def send_grpc(marker):

    audio_data,sr= soundfile.read("send_a2f.wav", dtype='float32')
                                                        
    if len(audio_data.shape) > 1:
        audio_data = np.average(audio_data, axis=1)
                                                            
    yield audio2face_pb2.PushAudioStreamRequest(start_marker=marker) 
    for i in range(len(audio_data) // sr//10 + 1):                  
        chunk = audio_data[i * sr//10: i * sr//10 + sr//10]
        yield audio2face_pb2.PushAudioStreamRequest(audio_data=chunk.astype(np.float32).tobytes())  
        
def send_stream():
                
 
    global audio_name
    global max_buffer 
    #global a2switch
    #global receive_flag
    global marker
    global send_ob
    send_ob=Avatar_instance_A
    #block_until_playback_is_finished = True  # ADJUST
    samplerate=44100
    audio_name="wangguan"
    #while True:
    with grpc.insecure_channel(a2f_url) as channel:
        stub= audio2face_pb2_grpc.Audio2FaceStub(channel)
        print("Channel start created success")
        while True:
      #
            print("------------------Channel restart created-------------------")
            marker = audio2face_pb2.PushAudioRequestStart(
                        samplerate=samplerate,
                        instance_name=send_ob,
                        block_until_playback_is_finished=True,
                    )
             
            def create_generator(a2switch): 
                                
                global send_ob 
                global audio_name
                global marker
                global predictor                   
                global receive_flag
                global max_buffer
                #global a2switch
                infer_buffer=[]
                while True:
                        
                        if a2switch: 
                            print("Send last audio!!!!!!!!!!!!!!")    
                            a2switch=False                                
                            send_grpc(marker)
                
                        elif (receive_flag) and len(max_buffer):
                                                    
                            send_buffer=[]  
                            send_buffer=max_buffer[:len(max_buffer)]
                            del max_buffer[:len(send_buffer)]
                                                    
                            infer_buffer.extend(send_buffer)
                                                    
                            wf = wave.open('send_a2f.wav', 'wb')   
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(p.get_sample_size(FORMAT))
                            wf.setframerate(RATE) 
                            wf.writeframes(b''.join(send_buffer))  
                            wf.close()
                                                                                                            
                            audio_data,sr= soundfile.read('send_a2f.wav', dtype='float32')
                                                        
                            if len(audio_data.shape) > 1:
                                audio_data = np.average(audio_data, axis=1)
                                                            
                            yield audio2face_pb2.PushAudioStreamRequest(start_marker=marker) 
                            for i in range(len(audio_data) // sr//10 + 1):
                                
                                chunk = audio_data[i * sr//10: i * sr//10 + sr//10]
                                yield audio2face_pb2.PushAudioStreamRequest(audio_data=chunk.astype(np.float32).tobytes())
                                                          
                        else: 
                            print("【INFO 】 waiting buffer ...")
                            continue

if __name__ == "__main__":
    
    text = "这里是一段较长的文本，需要拆分成多个句子来进行语音合成！句子也可以用问号来结尾吗？\
    当然可以。我要实现一个人工智能,这里是一段较长的文本，需要拆分成多个句子来进行语音合成！句子也可以用问号来结尾吗？当然可以。我要实现一个人工智能，但是我需要很多时间和精力完成\
        这里是一段较长的文本，需要拆分成多个句子来进行语音合成！句子也可以用问号来结尾吗？当然可以。我要实现一个人工智能，但是我需要很多时间和精力完成"
    # 启动主程序
    t1=time.time()
    asyncio.run(tts_send(text))
   
    
    print(time.time()-t1)
  #  t1 = threading.Thread(target=send_stream)
    t1=time.time()
    #asyncio.run(tts_a2f(text))
    print(time.time()-t1)
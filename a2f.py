import argparse
import functools
import os
import yaml
import numpy as np
import ffmpeg
import grpc
import grpc
import audio2face_pb2
import audio2face_pb2_grpc
from pydub import AudioSegment
from pydub.silence import split_on_silence
import soundfile
from audio2face_streaming_utils import push_audio_track_stream,push_audio_track,push_stream
import pyaudio
import wave
from queue import Queue
import time
import whisper 
import requests
#from llm_cards.bridge_chatgpt import predict
from config_private import API_KEY
import uuid
import re 
import asyncio
import threading
# 创建事件，用于线程间同步
send_event = threading.Event()   

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
                
                
async def tts_a2f(text):
    import edge_tts
    import soundfile as sf
    import numpy as np
    from  audio2face_streaming_utils import push_audio_track_stream
    generate_wave = edge_tts.Communicate(text, voice='zh-CN-YunxiNeural', rate='-5%', volume='+1%')
    await generate_wave.save('./voice_dir/send_frame.wav')  
                                       
    try:
        audio_data, samplerate = sf.read('./voice_dir/send_frame.wav', dtype="float32")
        if len(audio_data.shape) > 1:
            audio_data = np.average(audio_data, axis=1) 
        push_audio_track_stream(a2f_url, audio_data, samplerate , Avatar_instance_A) 
        print("send done")
        return 'Send Done!' 
    except Exception as e:
        print(f"检查是否开启omniverse!!!")              
              

def push_stream(url,player,dir="voice_dir/send_omniverse.wav"): 
    from audio2face_streaming_utils import push_audio_track_stream
    import soundfile    
    import numpy as np
    retry=0
    while True:  
        try:                    
            audio_data,sr= soundfile.read(dir, dtype='float32');break    
        except :
            print("tts合成速度稍慢，等待....")
            retry += 1
            print('正在重试')
            if retry >=2: raise TimeoutError                   
    if len(audio_data.shape) > 1:
        audio_data = np.average(audio_data, axis=1)    
    push_audio_track_stream(url, audio_data, sr, player)          
             
                
def audio_synthesis(gpt_replying_buffer,url,player):
    import threading
    threading.Thread(target=process_send_stream, args=(gpt_replying_buffer,url,player,)).start()
       
def process_send_stream(gpt_replying_buffer,url,player):
    import subprocess
    dir="voice_dir/send_omniverse.wav"
    cmd = f'edge-tts --voice {"zh-CN-YunxiNeural"} --text "{gpt_replying_buffer}" --write-media {dir}   '
    subprocess.run(cmd, shell=True) 
    time.sleep(0.5)
    push_stream(url,player,dir)
                
def receive_max(q,Text):
     
    global receive_flag 
    receive_flag=True
    sentences = re.split(r'[！？。: ,]', Text)
    sentences = [s.strip() for s in sentences if s.strip()] 
   # from VITS import 
    while True :
        
          if len(sentences)>0 :
               
               #audio_data=vit_tts(sentences.pop(0)
               #audio_data=r'voice_dir/send_frame.wav'
               audio_data=edge_tts.Communicate(sentences.pop(0), voice='zh-CN-YunxiNeural', rate='+1%', volume='+1%')
               q.put((audio_data,True))
               print('done')
          else :
               print('语音合成线程结束......')
               receive_flag=False  
               break 
      

###--------线程：收集数据，中转处理源buffer收集后发送------------###                           
def send_stream2(q):
    global mess      
    global receive_flag
    mess=False
    with grpc.insecure_channel(a2f_url) as channel:
        stub= audio2face_pb2_grpc.Audio2FaceStub(channel)    
        def create_generator():
            global mess
            while True:
                if not q.empty():       
                                #取出队列中的音频文件路径和对应的发送标志位
                                #print("检查缓存容量 :",q.qsize()) 
                                #time.sleep(2)
                                audio_data,send_flag = q.get()
                                if not send_flag:
                                    # TODO: 将音频文件发送出去
                                    print(f'Sending audio...')                               
                                    audio_data,sr= soundfile.read('voice_dir/send_framex.wav', dtype='float32')                      
                                    if len(audio_data.shape) > 1:
                                        audio_data = np.average(audio_data, axis=1)    
                                                            
                                    #yield audio2face_pb2.PushAudioStreamRequest(start_marker=Avatar_instance_A) 
                                    #for i in range(len(audio_data) // sr//10 + 1):      
                                                       # chunk = audio_data[i * sr//10: i * sr//10+ sr//10]
                                                        #yield audio2face_pb2.PushAudioStreamRequest(audio_data=chunk.astype(np.float32).tobytes()) 
                                    push_audio_track_stream(a2f_url, audio_data, sr, Avatar_instance_A)            
                                    send_flag=True
                                    # 重置事件状态
                                    send_event.clear()  
                                                                                                        
                else: 
                                if not receive_flag:
                                    print("发送线程结束")
                                    break 
                                else:
                                    continue
        stub.PushAudioStream(create_generator())                            

def audio_chatbot(text):
     
     q = Queue()
     
     t1 = threading.Thread(target=receive_max,args=(q,text))
     t2 = threading.Thread(target=send_stream2,args=(q,))
     t1.start()
     t2.start()
    # t1.join()
     #t2.join()
     global receive_flag
     while True:
          send_flag=True
    # 从队列中取出音频文件路径和对应的发送标志位
          audio, send_flag = q.get()
          if not send_flag:
               # 将音频文件路径放回队列（因为发送是在另一个线程中完成的）
               q.put((audio,False))
               # 设置事件，通知发送线程可以发送该音频
               send_event.set()           
          if not receive_flag:
               break
           
if __name__ == "__main__":
    
    text = "这里是一段较长的文本，需要拆分成多个句子来进行语音合成！句子也可以用问号来结尾吗？\
    当然可以。我要实现一个人工智能,这里是一段较长的文本，需要拆分成多个句子来进行语音合成！句子也可以用问号来结尾吗？当然可以。我要实现一个人工智能，但是我需要很多时间和精力完成\
        这里是一段较长的文本，需要拆分成多个句子来进行语音合成！句子也可以用问号来结尾吗？当然可以。我要实现一个人工智能，但是我需要很多时间和精力完成"
    # 启动主程序
    audio_chatbot(text)
#     t1=time.time()
#     asyncio.run(tts_send(text))
   
    
#     print(time.time()-t1)
#   #  t1 = threading.Thread(target=send_stream)
#     t1=time.time()
#     #asyncio.run(tts_a2f(text))
#     print(time.time()-t1)
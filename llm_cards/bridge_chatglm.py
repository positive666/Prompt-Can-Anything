
from transformers import AutoModel, AutoTokenizer
import time
import threading
import importlib
from utils.toolbox import update_ui, get_conf
from multiprocessing import Process, Pipe
import multiprocessing as mp
import torch 
load_message = "ChatGLM尚未加载，加载需要一段时间。注意，取决于`config.py`的配置，ChatGLM消耗大量的内存（CPU）或显存（GPU），也许会导致低配计算机卡死 ……"

#################################################################################
class GetGLMHandle(Process):
    def __init__(self,quantize='None'):
        super().__init__(daemon=True)
        self.parent, self.child = Pipe()
        self.chatglm_model = None
        self.chatglm_tokenizer = None
        self.info = ""
        mp.set_start_method('spawn')
        self.success = True
        self.quantize=quantize
        self.check_dependency()
        self.start()
        self.threadLock = threading.Lock()
        
    def check_dependency(self):
        try:
            import sentencepiece
            self.info = "依赖检测通过"
            self.success = True
        except:
            self.info = "缺少ChatGLM的依赖，如果要使用ChatGLM，除了基础的pip依赖以外，您还需要运行`pip install -r request_llm/requirements_chatglm.txt`安装ChatGLM的依赖。"
            self.success = False

    def ready(self):
        return self.chatglm_model is not None

    def run(self):
        # 子进程执行
        # 第一次运行，加载参数
        torch.cuda.init()
        retry = 0
       

        while True:
            try:
                if self.chatglm_model is None:
                    self.chatglm_tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True,cache_dir='weights')
                    device, = get_conf('LOCAL_MODEL_DEVICE')
                    if device=='cpu':
                        self.chatglm_model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True,cache_dir='weights').float()
                    else:
                        if int(self.quantize)==8 or int(self.quantize)==4:
                            self.chatglm_model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True,cache_dir='weights').quantize(int(self.quantize)).half().cuda()
                        else :
                            self.chatglm_model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True,cache_dir='weights').half().cuda()   
                    self.chatglm_model = self.chatglm_model.eval()
                    break
                else:
                    break
            except:
                retry += 1
                if retry > 3: 
                    self.child.send('[Local Message] Call ChatGLM fail 不能正常加载ChatGLM的参数。')
                    raise RuntimeError("不能正常加载ChatGLM的参数！")

        while True:
            # 进入任务等待状态
            kwargs = self.child.recv()
            # 收到消息，开始请求
            try:
                for response, history in self.chatglm_model.stream_chat(self.chatglm_tokenizer, **kwargs):
                    self.child.send(response)
                    # # 中途接收可能的终止指令（如果有的话）
                    # if self.child.poll(): 
                    #     command = self.child.recv()
                    #     if command == '[Terminate]': break
            except:
                from utils.toolbox import trimmed_format_exc
                self.child.send('[Local Message] Call ChatGLM fail.' + '\n```\n' + trimmed_format_exc() + '\n```\n')
            # 请求处理结束，开始下一个循环
            self.child.send('[Finish]')

    def stream_chat(self, **kwargs):
        # 主进程执行
        self.threadLock.acquire()
        self.parent.send(kwargs)
        while True:
            res = self.parent.recv()
            if res != '[Finish]':
                yield res
            else:
                break
        self.threadLock.release()
    
global glm_handle

glm_handle = None
#################################################################################
def predict_no_ui_long_connection(inputs, llm_kwargs, history=[], sys_prompt="", observe_window=[], console_slience=False):
    """
        多线程方法
        函数的说明请见 request_llm/bridge_all.py
    """
    global glm_handle
    if glm_handle is None:
        glm_handle = GetGLMHandle()
        if len(observe_window) >= 1: observe_window[0] = load_message + "\n\n" + glm_handle.info
        if not glm_handle.success: 
            error = glm_handle.info
            glm_handle = None
            raise RuntimeError(error)

    # chatglm 没有 sys_prompt 接口，因此把prompt加入 history
    history_feedin = []
    history_feedin.append(["What can I do?", sys_prompt])
    for i in range(len(history)//2):
        history_feedin.append([history[2*i], history[2*i+1]] )

    watch_dog_patience = 5 # 看门狗 (watchdog) 的耐心, 设置5秒即可
    response = ""
    for response in glm_handle.stream_chat(query=inputs, history=history_feedin, max_length=llm_kwargs['max_length'], top_p=llm_kwargs['top_p'], temperature=llm_kwargs['temperature']):
        if len(observe_window) >= 1:  observe_window[0] = response
        if len(observe_window) >= 2:  
            if (time.time()-observe_window[1]) > watch_dog_patience:
                raise RuntimeError("程序终止。")
    return response

import threading 
import queue

def asr_with_gpt(transcriber,text_queue, llm_kwargs, plugin_kwargs, chatbot, history=[], system_prompt='', stream = True, additional_fn=None):
    print('开启识别线程')
   
    while True:
            if transcriber.transcript_changed_event.is_set():
                start_time = time.time()
                transcriber.transcript_changed_event.clear() 
                transcript_string=''
                if transcriber.two_ways:
                    transcript_string = transcriber.get_transcript()
                else:
                    #print(transcriber.transcript_data["You"])
                    transcript_string=transcriber.transcript_data["You"] 
                    #transcript_string=transcript_string[0][0][len('You: [') : -len(']\n\n')]
                if transcript_string!=" ":
                    response=predict_no_ui_long_connection(str(transcript_string), llm_kwargs, history, system_prompt, None,False) 
                    end_time = time.time()  # Measure end time
                    execution_time = end_time - start_time  # Calculate the time it took to execute the function
                    
                    text_queue.put((transcript_string,response))
                    remaining_time = 2 - execution_time
                    if remaining_time > 0:
                        time.sleep(remaining_time)
            
            else:
                time.sleep(0.02) # 允许事件循环在这里运行，避免耗费 CPU                     
        

async def send_ui(text_queue,chatbot, history): 
    #time.sleep(0.01)
    while  True:
       #     print('updata ui',text_queue.qsize())
        if text_queue.qsize()>0:
                iss,ans=text_queue.get()
                chatbot.append((iss,ans))
                
                history.extend([iss,ans])
                
                print('更新界面中。。。。。')
             
                await chatbot.get_cookies(),chatbot, history, 'done'
        else:
            print('队列为空')
        
def Talk_with_app(inputs, llm_kwargs, plugin_kwargs, chatbot, history=[], system_prompt='', stream = True, additional_fn=None):
        import threading 
        from utils import AudioRecorder 
        from utils import AudioTrans
        import whisper
        
        chat_app=llm_kwargs.get('chat_app', False)
        print(f'是否开启自动对话系统：{chat_app}')
        model = whisper.load_model('small',download_root='weights')  
        # 在主线程中初始化 loop
        #loop = asyncio.get_event_loop() 
        
        if(chat_app):
            chatbot.append(("开启自动对话系统", "等待效应开启语音识别服务"))
            #history.append("")
            yield from update_ui(chatbot=chatbot, history=history, msg="等待响应") # 刷新界面
            audio_queue = queue.Queue()
            text_queue=queue.Queue()
            user_audio_recorder = AudioRecorder.DefaultMicRecorder()
            user_audio_recorder.record_into_queue(audio_queue)

            time.sleep(2)
            #speaker_audio_recorder = AudioRecorder.DefaultSpeakerRecorder()
            #speaker_audio_recorder.record_into_queue(audio_queue)
            print('开始语音识别服务........')
        
            transcriber = AudioTrans.AudioTranscriber(user_audio_recorder.source, None, model)
            transcribe = threading.Thread(target=transcriber.transcribe_audio_queue, args=(audio_queue,))
            transcribe.daemon=True
            transcribe.start()
            t1=threading.Thread(target=asr_with_gpt, args=(transcriber, text_queue,llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, stream , additional_fn,))
            t1.daemon=True
            t1.start()
        
            while  True:              
                while text_queue.qsize()>0:
                        try:
                            iss,ans=text_queue.get()
                            chatbot.append((iss,ans))                   
                            history.extend([iss,ans])

                           # print('更新界面中。。。。。')
                            yield from update_ui(chatbot=chatbot, history=history, msg="Done") 
                        except Exception as e:  
                            yield from update_ui(chatbot=chatbot, history=history, msg="Error") 
                            return 
                time.sleep(0.02)  
                yield from update_ui(chatbot=chatbot, history=history, msg="等待响应") # 刷新界面
             
        chatbot[-1]=('退出对话系统',"已结束对话系统")   
        yield from update_ui(chatbot=chatbot, history=history, msg="远程返回:") # 刷新界面

def predict(inputs, llm_kwargs, plugin_kwargs, chatbot, history=[], system_prompt='', stream = True, additional_fn=None):
    """
        单线程方法
        函数的说明请见 request_llm/bridge_all.py
    """
    chatbot.append((inputs, ""))
    omniverse_state = llm_kwargs.get('omniverse', False)
    record_file=llm_kwargs.get('record_audio', False)
    asr=llm_kwargs.get('asr', False)
    if record_file and asr:
        import whisper
        from a2f import speech_recognition
        speech_text,speech_language=speech_recognition(record_file,whisper.load_model("small",
                                download_root="weights") ,False)                             
        inputs=speech_text  
        print('asr result:',inputs)   
    
    global glm_handle
    quantize=llm_kwargs.get('quantize', False)  
    print(f'准备进行量化{quantize}')
    if glm_handle is None:
        glm_handle = GetGLMHandle(quantize=quantize)
        chatbot[-1] = (inputs, load_message + "\n\n" + glm_handle.info)
        yield from update_ui(chatbot=chatbot, history=[])
        if not glm_handle.success: 
            glm_handle = None
            return

    if additional_fn is not None:
        import core_functional
        importlib.reload(core_functional)    # 热更新prompt
        core_functional = core_functional.get_core_functions()
        if "PreProcess" in core_functional[additional_fn]: inputs = core_functional[additional_fn]["PreProcess"](inputs)  # 获取预处理函数（如果有的话）
        inputs = core_functional[additional_fn]["Prefix"] + inputs + core_functional[additional_fn]["Suffix"]

    # 处理历史信息
    history_feedin = []
    history_feedin.append(["What can I do?", system_prompt] )
    for i in range(len(history)//2):
        history_feedin.append([history[2*i], history[2*i+1]] )

    # 开始接收chatglm的回复
    response = "[Local Message]: 等待ChatGLM响应中 ..."
    for response in glm_handle.stream_chat(query=inputs, history=history_feedin, max_length=llm_kwargs['max_length'], top_p=llm_kwargs['top_p'], temperature=llm_kwargs['temperature']):
        chatbot[-1] = (inputs, response)
        yield from update_ui(chatbot=chatbot, history=history)

    # 总结输出
    if response == "[Local Message]: 等待ChatGLM响应中 ...":
        response = "[Local Message]: ChatGLM响应异常 ..."
    history.extend([inputs, response])
    yield from update_ui(chatbot=chatbot, history=history)

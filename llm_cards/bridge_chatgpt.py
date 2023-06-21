# 借鉴了 https://github.com/GaiZhenbiao/ChuanhuChatGPT 项目

"""
    该文件中主要包含三个函数

    不具备多线程能力的函数：
    1. predict: 正常对话时使用，具备完备的交互功能，不可多线程

    具备多线程调用能力的函数
    2. predict_no_ui：高级实验性功能模块调用，不会实时显示在界面上，参数简单，可以多线程并行，方便实现复杂的功能逻辑
    3. predict_no_ui_long_connection：在实验过程中发现调用predict_no_ui处理长文档时，和openai的连接容易断掉，这个函数用stream的方式解决这个问题，同样支持多线程
"""

import json
import time
import gradio as gr
import logging
import traceback
import requests
import importlib
from a2f import audio_synthesis
# config_private.py放自己的秘密如API和代理网址
# 读取时首先看是否存在私密的config_private配置文件（不受git管控），如果有，则覆盖原config文件
from utils.toolbox import get_conf, update_ui, is_any_api_key, select_api_key, what_keys, clip_history, trimmed_format_exc
proxies, API_KEY, TIMEOUT_SECONDS, MAX_RETRY,A2F_URL,Avatar_instance_A = \
    get_conf('proxies', 'API_KEY', 'TIMEOUT_SECONDS', 'MAX_RETRY','A2F_URL','Avatar_instance_A')

timeout_bot_msg = '[Local Message] Request timeout. Network error. Please check proxy settings in config.py.' + \
                  '网络错误，检查代理服务器是否可用，以及代理设置的格式是否正确，格式须是[协议]://[地址]:[端口]，缺一不可。'

def get_full_error(chunk, stream_response):
    """
        获取完整的从Openai返回的报错
    """
    while True:
        try:
            chunk += next(stream_response)
        except:
            break
    return chunk


def predict_no_ui_long_connection(inputs, llm_kwargs, history=[], sys_prompt="", observe_window=None, console_slience=False):
    """
    发送至chatGPT，等待回复，一次性完成，不显示中间过程。但内部用stream的方法避免中途网线被掐。
    inputs：
        是本次问询的输入
    sys_prompt:
        系统静默prompt
    llm_kwargs：
        chatGPT的内部调优参数
    history：
        是之前的对话列表
    observe_window = None：
        用于负责跨越线程传递已经输出的部分，大部分时候仅仅为了fancy的视觉效果，留空即可。observe_window[0]：观测窗。observe_window[1]：看门狗
    """
    watch_dog_patience = 5 # 看门狗的耐心, 设置5秒即可
    headers, payload = generate_payload(inputs, llm_kwargs, history, system_prompt=sys_prompt, stream=True)
    retry = 0
    while True:
        try:
            # make a POST request to the API endpoint, stream=False
            from llm_cards.bridge_all import model_info
            endpoint = model_info[llm_kwargs['llm_model']]['endpoint']
            response = requests.post(endpoint, headers=headers, proxies=proxies,
                                    json=payload, stream=True, timeout=TIMEOUT_SECONDS); break
        except requests.exceptions.ReadTimeout as e:
            retry += 1
            traceback.print_exc()
            if retry > MAX_RETRY: raise TimeoutError
            if MAX_RETRY!=0: print(f'请求超时，正在重试 ({retry}/{MAX_RETRY}) ……')

    stream_response =  response.iter_lines()
    result = ''
    while True:
        try: chunk = next(stream_response).decode()
        except StopIteration: 
            break
        except requests.exceptions.ConnectionError:
            chunk = next(stream_response).decode() # 失败了，重试一次？再失败就没办法了。
        if len(chunk)==0: continue
        if not chunk.startswith('data:'): 
            error_msg = get_full_error(chunk.encode('utf8'), stream_response).decode()
            if "reduce the length" in error_msg:
                raise ConnectionAbortedError("OpenAI拒绝了请求:" + error_msg)
            else:
                raise RuntimeError("OpenAI拒绝了请求：" + error_msg)
        if ('data: [DONE]' in chunk): break # api2d 正常完成
        json_data = json.loads(chunk.lstrip('data:'))['choices'][0]
        delta = json_data["delta"]
        if len(delta) == 0: break
        if "role" in delta: continue
        if "content" in delta: 
            result += delta["content"]
            if not console_slience: print(delta["content"], end='')
            if observe_window is not None: 
                # 观测窗，把已经获取的数据显示出去
                if len(observe_window) >= 1: observe_window[0] += delta["content"]
                # 看门狗，如果超过期限没有喂狗，则终止
                if len(observe_window) >= 2:  
                    if (time.time()-observe_window[1]) > watch_dog_patience:
                        raise RuntimeError("用户取消了程序。")
        else: raise RuntimeError("意外Json结构："+delta)
    if json_data['finish_reason'] == 'length':
        raise ConnectionAbortedError("正常结束，但显示Token不足，导致输出不完整，请削减单次输入的文本量。")
    return result

global asr_handle
asr_handle=None
import queue
def asr_with_gpt(transcriber,text_queue, llm_kwargs, plugin_kwargs, chatbot, history=[], system_prompt='', stream = True, additional_fn=None):
    print('开启识别线程')
    history_text=''
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
                if transcript_string!=" "and transcript_string!= history_text :
                    response=predict_no_ui_long_connection(str(transcript_string), llm_kwargs, history, system_prompt, None,False) 
                    end_time = time.time()  # Measure end time
                    execution_time = end_time - start_time  # Calculate the time it took to execute the function
                    
                    text_queue.put((transcript_string,response))
                    remaining_time = 2 - execution_time
                    if remaining_time > 0:
                        time.sleep(remaining_time)
            
            else:
                time.sleep(0.02) # 允许事件循环在这里运行，避免耗费 CPU                     
        
def Talk_with_app(inputs, llm_kwargs, plugin_kwargs, chatbot, history=[], system_prompt='', stream = True, additional_fn=None):
        import threading 
        from utils import AudioRecorder 
        from utils import AudioTrans
        import whisper
        
        chat_app=llm_kwargs.get('chat_app', False)
        print(f'是否开启自动对话系统：{chat_app}')
        model = whisper.load_model('small',download_root='weights')  
        
        if(chat_app):
            chatbot.append(("开启自动对话系统", "等待效应开启语音识别服务"))
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
            
            while True:              
                while text_queue.qsize()>0:
                        try:
                            iss,ans=text_queue.get()
                            chatbot.append((iss,ans))                   
                            history.extend([iss,ans])
                            audio_synthesis(ans,A2F_URL,Avatar_instance_A)
                            yield from update_ui(chatbot=chatbot, history=history, msg="Done") 
                        except Exception as e:
                            chatbot[-1][-1]='超时错误'  
                            print(e)
                            yield from update_ui(chatbot=chatbot, history=history, msg="Error") 
                            return 
                time.sleep(0.01)  
                yield from update_ui(chatbot=chatbot, history=history, msg="等待响应") # 刷新界面
        
        chatbot[-1]=('退出对话系统',"已结束对话系统")   
        yield from update_ui(chatbot=chatbot, history=history, msg="远程返回:") # 刷新界面
    
    
# async def Talk_with_app(inputs, llm_kwargs, plugin_kwargs, chatbot, history=[], system_prompt='', stream = True, additional_fn=None):
#     await asyncio.gather(talk_with_app(inputs, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, stream, additional_fn), send_ui )
def predict(inputs, llm_kwargs, plugin_kwargs, chatbot, history=[], system_prompt='', stream = True, additional_fn=None):
    """
    发送至chatGPT，流式获取输出。
    用于基础的对话功能。
    inputs 是本次问询的输入
    top_p, temperature是chatGPT的内部调优参数
    history 是之前的对话列表（注意无论是inputs还是history，内容太长了都会触发token数量溢出的错误）
    chatbot 为WebUI中显示的对话列表，修改它，然后yeild出去，可以直接修改对话界面内容
    additional_fn代表点击的哪个按钮，按钮见functional.py
    """
    omniverse_state = llm_kwargs.get('omniverse', False)
    record_file=llm_kwargs.get('record_audio', False)
    asr=llm_kwargs.get('asr', False)
    if asr:     
        import whisper
        global asr_handle    
        if asr_handle is  None:
            print('loads asf model..')
            asr_handle=whisper.load_model("small",
                                 download_root="weights")
           
        result = asr_handle.transcribe(record_file)
        inputs=result["text"]
        print('result:',inputs) 
     
    if is_any_api_key(inputs):
        chatbot._cookies['api_key'] = inputs
        chatbot.append(("输入已识别为openai的api_key", what_keys(inputs)))
        yield from update_ui(chatbot=chatbot, history=history, msg="api_key已导入") # 刷新界面
        return
    elif not is_any_api_key(chatbot._cookies['api_key']):
        chatbot.append((inputs, "缺少api_key。\n\n1. 临时解决方案：直接在输入区键入api_key，然后回车提交。\n\n2. 长效解决方案：在config.py中配置。"))
        yield from update_ui(chatbot=chatbot, history=history, msg="缺少api_key") # 刷新界面
        return
    
    if additional_fn is not None:
        from llm_cards import core_functional
        importlib.reload(core_functional)    # 热更新prompt
        core_functional = core_functional.get_core_functions()
        if "PreProcess" in core_functional[additional_fn]: inputs = core_functional[additional_fn]["PreProcess"](inputs)  # 获取预处理函数（如果有的话）
        inputs = core_functional[additional_fn]["Prefix"] + inputs + core_functional[additional_fn]["Suffix"]
  
        
    raw_input = inputs
    logging.info(f'[raw_input] {raw_input}')
    chatbot.append((inputs, ""))
    yield from update_ui(chatbot=chatbot, history=history, msg="等待响应") # 刷新界面

    try:
        headers, payload = generate_payload(inputs, llm_kwargs, history, system_prompt, stream)
    except RuntimeError as e:
        chatbot[-1] = (inputs, f"您提供的api-key不满足要求，不包含任何可用于{llm_kwargs['llm_model']}的api-key。您可能选择了错误的模型或请求源。")
        yield from update_ui(chatbot=chatbot, history=history, msg="api-key不满足要求") # 刷新界面
        return
        
    history.append(inputs); history.append("")
    retry = 0
    while True:
        try:
            # make a POST request to the API endpoint, stream=True
            from llm_cards.bridge_all import model_info
            endpoint = model_info[llm_kwargs['llm_model']]['endpoint']
            response = requests.post(endpoint, headers=headers, proxies=proxies,
                                    json=payload, stream=True, timeout=TIMEOUT_SECONDS);break
        except:
            retry += 1
            chatbot[-1] = ((chatbot[-1][0], timeout_bot_msg))
            retry_msg = f"，正在重试 ({retry}/{MAX_RETRY}) ……" if MAX_RETRY > 0 else ""
            yield from update_ui(chatbot=chatbot, history=history, msg="请求超时"+retry_msg) # 刷新界面
            if retry > MAX_RETRY: raise TimeoutError

    gpt_replying_buffer = ""
    count=0
    is_head_of_the_stream = True
    if stream:
        stream_response =  response.iter_lines()
        while True:
            try:
                chunk = next(stream_response)
            except StopIteration:
                # 非OpenAI官方接口的出现这样的报错，OpenAI和API2D不会走这里
                from utils.toolbox import regular_txt_to_markdown; tb_str = '```\n' + trimmed_format_exc() + '```'
                chatbot[-1] = (chatbot[-1][0], f"[Local Message] 远程返回错误: \n\n{tb_str} \n\n{regular_txt_to_markdown(chunk.decode())}")
                yield from update_ui(chatbot=chatbot, history=history, msg="远程返回错误:" + chunk.decode()) # 刷新界面
                return
            
            if is_head_of_the_stream and (r'"object":"error"' not in chunk.decode()):
                # 数据流的第一帧不携带content
                is_head_of_the_stream = False; continue
            
            if chunk:
                try:
                    chunk_decoded = chunk.decode()
                    # 前者API2D的
                    if ('data: [DONE]' in chunk_decoded) or (len(json.loads(chunk_decoded[6:])['choices'][0]["delta"]) == 0):
                        # 判定为数据流的结束，gpt_replying_buffer也写完了
                        logging.info(f'[response] {gpt_replying_buffer}')
                        break
                    # 处理数据流的主体
                    chunkjson = json.loads(chunk_decoded[6:])
                    status_text = f"finish_reason: {chunkjson['choices'][0]['finish_reason']}"
                    # 如果这里抛出异常，一般是文本过长，详情见get_full_error的输出
                    new_buffer=json.loads(chunk_decoded[6:])['choices'][0]["delta"]["content"]
                    gpt_replying_buffer = gpt_replying_buffer + new_buffer
                    history[-1] = gpt_replying_buffer
                    # if omniverse_state and gpt_replying_buffer[-1] in('。'):

                    chatbot[-1] = (history[-2], history[-1])                  
             
                    yield from update_ui(chatbot=chatbot, history=history, msg=status_text) # 刷新界面
                    
                
                except Exception as e:
                    traceback.print_exc()
                    yield from update_ui(chatbot=chatbot, history=history, msg="Json解析不合常规") # 刷新界面
                    chunk = get_full_error(chunk, stream_response)
                    chunk_decoded = chunk.decode()
                    error_msg = chunk_decoded
                    if "reduce the length" in error_msg:
                        if len(history) >= 2: history[-1] = ""; history[-2] = "" # 清除当前溢出的输入：history[-2] 是本次输入, history[-1] 是本次输出
                        history = clip_history(inputs=inputs, history=history, tokenizer=model_info[llm_kwargs['llm_model']]['tokenizer'], 
                                               max_token_limit=(model_info[llm_kwargs['llm_model']]['max_token'])) # history至少释放二分之一
                        chatbot[-1] = (chatbot[-1][0], "[Local Message] Reduce the length. 本次输入过长, 或历史数据过长. 历史缓存数据已部分释放, 您可以请再次尝试. (若再次失败则更可能是因为输入过长.)")
                        # history = []    # 清除历史
                    elif "does not exist" in error_msg:
                        chatbot[-1] = (chatbot[-1][0], f"[Local Message] Model {llm_kwargs['llm_model']} does not exist. 模型不存在, 或者您没有获得体验资格.")
                    elif "Incorrect API key" in error_msg:
                        chatbot[-1] = (chatbot[-1][0], "[Local Message] Incorrect API key. OpenAI以提供了不正确的API_KEY为由, 拒绝服务.")
                    elif "exceeded your current quota" in error_msg:
                        chatbot[-1] = (chatbot[-1][0], "[Local Message] You exceeded your current quota. OpenAI以账户额度不足为由, 拒绝服务.")
                    elif "bad forward key" in error_msg:
                        chatbot[-1] = (chatbot[-1][0], "[Local Message] Bad forward key. API2D账户额度不足.")
                    elif "Not enough point" in error_msg:
                        chatbot[-1] = (chatbot[-1][0], "[Local Message] Not enough point. API2D账户点数不足.")
                    else:
                        from utils.toolbox import regular_txt_to_markdown
                        tb_str = '```\n' + trimmed_format_exc() + '```'
                        chatbot[-1] = (chatbot[-1][0], f"[Local Message] 异常 \n\n{tb_str} \n\n{regular_txt_to_markdown(chunk_decoded)}")
                    yield from update_ui(chatbot=chatbot, history=history, msg="Json异常" + error_msg) # 刷新界面
                    return
           
        if omniverse_state:
            from a2f import tts_a2f
            audio_synthesis(chatbot[-1][-1],A2F_URL,Avatar_instance_A)   
           #asyncio.run(tts_a2f(chatbot[-1][-1]))

        
def generate_payload(inputs, llm_kwargs, history, system_prompt, stream):
    """
    整合所有信息，选择LLM模型，生成http请求，为发送请求做准备
    """
    if not is_any_api_key(llm_kwargs['api_key']):
        raise AssertionError("你提供了错误的API_KEY。\n\n1. 临时解决方案：直接在输入区键入api_key，然后回车提交。\n\n2. 长效解决方案：在config.py中配置。")

    api_key = select_api_key(llm_kwargs['api_key'], llm_kwargs['llm_model'])

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    conversation_cnt = len(history) // 2

    messages = [{"role": "system", "content": system_prompt}]
    if conversation_cnt:
        for index in range(0, 2*conversation_cnt, 2):
            what_i_have_asked = {}
            what_i_have_asked["role"] = "user"
            what_i_have_asked["content"] = history[index]
            what_gpt_answer = {}
            what_gpt_answer["role"] = "assistant"
            what_gpt_answer["content"] = history[index+1]
            if what_i_have_asked["content"] != "":
                if what_gpt_answer["content"] == "": continue
                if what_gpt_answer["content"] == timeout_bot_msg: continue
                messages.append(what_i_have_asked)
                messages.append(what_gpt_answer)
            else:
                messages[-1]['content'] = what_gpt_answer['content']

    what_i_ask_now = {}
    what_i_ask_now["role"] = "user"
    what_i_ask_now["content"] = inputs
    messages.append(what_i_ask_now)

    payload = {
        "model": llm_kwargs['llm_model'].strip('api2d-'),
        "messages": messages, 
        "temperature": llm_kwargs['temperature'],  # 1.0,
        "top_p": llm_kwargs['top_p'],  # 1.0,
        "n": 1,
        "stream": stream,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }
    try:
        print(f" {llm_kwargs['llm_model']} : {conversation_cnt} : {inputs[:100]} ..........")
    except:
        print('输入中可能存在乱码。')
    return headers,payload



from model_cards.autoback import AutoBackend
import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np 
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
from utils.ops import (LOGGER, Profile, check_file, check_requirements, colorstr, cv2,
                     dilate_mask, increment_path , scale_boxes, xyxy2xywh,save_format)
from utils.plot import Annotator, save_one_box,show_box,show_mask,save_mask_data,Draw_img
from config_private import *
#from llm_cards.bridge_chatgpt import predict
from llm_cards.bridge_all import predict
from utils.toolbox import format_io, find_free_port, on_file_uploaded, on_report_generated, get_conf, ArgsGeneralWrapper, DummyWith

from utils.torch_utils import select_device
from config_private import SAM_MODEL_TYPE,GROUNED_MODEL_TYPE,Tag2Text_Model_Path
from utils import VID_FORMATS,IMG_FORMATS,write_categories
sys.path.append("VisualGLM")
from VisualGLM_6B.chatglm import  *
from a2f import *
#import xml.etree.cElementTree as ET
import gradio as gr
#from gradio.inputs import File
import random
import threading 
import asyncio
import concurrent.futures
from utils.colorful import *



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  #  root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import asyncio
from concurrent.futures import ThreadPoolExecutor
global categories
categories = {}
global category_colors
category_colors={}
# 初始对应类别编号
class_ids = []
global speech_AI
speech_AI={'whisper':None }

global models_config
models_config = {'tag2text': None, 'lama': None,'sam': None,'grounded': None,'sd': None, 
                 'visual_glm': None , 'trans_zh': None,'gligen': None}

from llm_cards.core_functional import get_core_functions
functional = get_core_functions()
JSON_DATASETS=[]

async def load_speech_model(whisper=None):
        import whisper 
        global speech_AI
        if whisper:
            speech_AI['whisper'] =  whisper.load_model("small",download_root="weights")
            LOGGER.info('loads whisper')
            
        elif not whisper:
          LOGGER.info('free  memory')
          models_config['visual_glm']=None  
        else:    
          LOGGER.info('pass')         
        #await asyncio.sleep(0.01) 
        return '语音识别记载完成'
           
def save_text2img_data(prompt,label,img_name):
    global JSON_DATASETS
    if not prompt:
        prompt=f"这张图片的背景里有什么内容?"
        
    example = {
        "img": f"{img_name}",
        "prompt": prompt,
        "label": label
    }
    JSON_DATASETS.append(example)
            

def auto_opentab_delay(port=7586):
        import threading, webbrowser, time
        LOGGER.info(f"\n如果浏览器没有自动打开，请复制并转到以下URL：")
        LOGGER.info(f"\t（亮色主题）: http://localhost:{port}")
        LOGGER.info(f"\t（暗色主题）: http://localhost:{port}/?__theme=dark")
        def open():
            time.sleep(2)       # 打开浏览器
            DARK_MODE, = get_conf('DARK_MODE')
            if DARK_MODE: webbrowser.open_new_tab(f"http://localhost:{port}/?__theme=dark")
            else: webbrowser.open_new_tab(f"http://localhost:{port}")
        threading.Thread(target=open, name="open-browser", daemon=True).start()
        #threading.Thread(target=auto_update, name="self-upgrade", daemon=True).start()
        #threading.Thread(target=warm_up_modules, name="warm-up", daemon=True).start()


async def load_auto_backend_models(lama, sam, det, tag2text, trans_zh, visual_glm,device=0, quant=4, bar=None): 
    try:    
        with concurrent.futures.ThreadPoolExecutor() as pool:
                wait_coros =  asyncio.get_event_loop().run_in_executor(pool, load_auto_backend_model, lama, sam, det, tag2text, trans_zh, visual_glm,device, quant, bar)
                await asyncio.wait([wait_coros])
        await asyncio.sleep(0.01) 
    except Exception as e:
        LOGGER.info("An error occurred: ", e)
        return '第一次可能会出现问题,请再次点击加载按钮，也可以检查后台'
    return 'Loads Done !'
   

def load_auto_backend_model(lama,sam,det,tag2text,trans_zh,visual_glm,device,quant,bar):
    """
    加载模型库
    """
    # Load model    
    
    global models_config    
 
    if visual_glm and not models_config['visual_glm']:
         
          models_config['visual_glm']=VisualGLM(gpu_device=int(device),quant=int(quant))
          LOGGER.info(f'GPU{int(device)}———量化VisualGLM模型:int{int(quant)}')
    elif not visual_glm:
          LOGGER.info('free visualGLM memory')
          models_config['visual_glm']=None  
    else:    
          LOGGER.info('free or no visual_glm')      
           
    device = select_device(device)    
    if tag2text and not models_config['tag2text']:
                #progress.update(5) # 更新进度条
                models_config['tag2text'] = AutoBackend("tag2text",weights=Tag2Text_Model_Path,device=device)
               
    elif not tag2text  :
            LOGGER.info('free memory')
            models_config['tag2text'] =None 
    else :
            LOGGER.info('free or tag2text pass')   
             
    if det and not models_config['grounded']:
            models_config['grounded'] = AutoBackend("grounded-DINO",weights=GROUNED_MODEL_TYPE['S'], device=device,
            args_config= 'model_cards/groundingdino/config/GroundingDINO_SwinT_OGC.py')
            #progress.tqdm.write(f"{i+2}/{len(models_config)}")
    elif not det  :
            models_config['grounded'] =None 
    else :
            LOGGER.info('free or grounded pass')
            
    if sam and not models_config['sam']:
            models_config['sam']= AutoBackend("segment-anything",weights=SAM_MODEL_TYPE['vit_h'] ,device=device)
            #progress.tqdm.write(f"{i+3}/{len(models_config)}")
    elif not sam :
            models_config['sam'] =None      
    else:
            LOGGER.info("PASS SAM")
            
    if trans_zh and not models_config['trans_zh']:
            from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
            cn_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh",cache_dir='weights')
            cn_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh",cache_dir='weights')
            translator = pipeline("text2text-generation", model=cn_model, tokenizer=cn_tokenizer)
            models_config['trans_zh']= translator       
    elif not trans_zh  :
            models_config['trans_zh'] =None 
    else :
            LOGGER.info('zh model pass')    
        
    if lama and not models_config['lama']:
            models_config['lama']= AutoBackend("lama",weights=None,args_config='model_cards/lama/configs/prediction/default.yaml',device=device)
    elif not lama :
            models_config['lama'] =None 
    else :
            LOGGER.info('free or lama pass') 
   
    return 'OK'
    
def Auto_run(
        source= 'data/images',  # file/dir/URL/glob, 0 for webcam
        img_input='',
        input_prompt="Anything in this image",
        conf_thres=0.3,  # confidence threshold
        iou_thres=0.5,  # NMS IOU threshold
        text_thres=0.2,
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu   
        quant=4,
        save_conf=False,  # save confidences in --save-txt labels
        img_save=False,  # do not save images/videos
        visualize=False,  # visualize features
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        lama=False,   # use lama models
        sam=True,    # use segment-anythings
        det=True,    # use grounded detect model with text
        tag2text=True,
       # chatgpt=False,
        save_txt=False,  # save results to *.txt
        save_xml=False,  # save results to *.xml
        save_mask=False,
        save_caption=False,
        batch_process=False,
        color_flag=False,
        zh_select=False,
        record_audio=None,
        up_audio=None,
        process_name=0,
        
        ):  
            
            global models_config
            global category_colors
            global JSON_DATASETS
           # load_auto_backend_models(lama,sam,det,tag2text,zh_select,device,quant)
            #LOGGER.info (f'proceess ID：{process_name},loads model list ：{models_config.keys()}')            
            cls_index = -1        # 设置默认值为 -1
            if img_input:
                source =img_input
            source = str(source)
     
            img_paths=None
            if os.path.isdir(source):
                img_paths = [os.path.join(source, f) for f in os.listdir(source) if
                    Path(f).suffix[1:] in (IMG_FORMATS + VID_FORMATS)] 
            else:
                img_paths = [source]    

            # Directories
            is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
          #  save_img = img_save and not source.endswith('.txt')  # save inference images
            is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
            #webcam = source.isnumeric() or source.endswith('.streams') or (is_url )
            if is_url and is_file:
                source = check_file(source)  # download

            save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
            
            (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
            (save_dir / 'xmls' if save_xml else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
            (save_dir / 'masks' if save_mask else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
            (save_dir / 'captions' if save_caption else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
            p = Path(str(save_dir) )  # to Path
            seen=0
            # loda data and inference
            caption=None
            for source in (img_paths): 
                    im = cv2.imread(source)
                    name_p= source.split('/')[-1].split('.')[0]
                    img_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    preds=None
                    masks=[]
                    prompt=input_prompt
                    if tag2text:
                        LOGGER.info(f'text_prompt:{prompt}')
                        preds = models_config['tag2text'](im = img_rgb ,prompt=prompt,box_threshold=conf_thres,text_threshold=text_thres,iou_threshold=iou_thres)
                    # Currently ", " is better for detecting single tags
                    # while ". " is a little worse in some case
                        prompt=preds[0].replace(' |', ',')
                        caption=preds[2]
                        LOGGER.info(f"Caption: {caption}")
                        LOGGER.info(f"Tags: {prompt}")
                        if zh_select and prompt :
                            caption=models_config['trans_zh'](caption, max_length=1000, clean_up_tokenization_spaces=True)[0]["generated_text"]
                        if save_caption:
                            save_text2img_data(None, caption,name_p)
                            #save_format(label_format="txt",save_path=f'{save_dir}/captions',img_name=name_p, results=caption)
                      
                    if det:
                        if input_prompt:
                            prompt=input_prompt
                            LOGGER.info('your input prompt replace default:',prompt)
                        preds= models_config['grounded'](im = img_rgb,prompt=prompt, box_threshold=conf_thres,text_threshold=text_thres, iou_threshold=iou_thres) 
                      
                    if sam and det :        
                        if preds[0].numel()>0:      
                            masks= models_config['sam'](im = img_rgb, prompt=preds[0],box_threshold=conf_thres,text_threshold=text_thres, iou_threshold=iou_thres)
                            if save_mask:
                                save_mask_data(str(save_dir)+'/masks', caption, masks, preds[0], preds[2],name_p)
                    # Write results
                    
                    if img_save:
                        seen+=1
                        plt.figure(figsize=(20,18))
                        plt.imshow(img_rgb)
                        if det:
                            for box,label in zip(preds[0],preds[2]):
                                    show_box(box.numpy(),plt.gca(),label)            
                            if sam :              
                                for mask in masks:         
                                    show_mask(mask.cpu().numpy(),plt.gca(),random_color=True)
                        if tag2text:
                            plt.title('Captioning: ' + caption + '\n' + 'Tagging:' + prompt + '\n')    
                        plt.axis('off')
                        plt.savefig(f'{save_dir}/{seen}.jpg',bbox_iches='tight',dpi=600,pad_inches=0.0)     
                        
                    if lama and masks is not None :      
                        masks_prompts= masks.detach().cpu().numpy().astype(np.uint8) * 255
                        for idx, mask in enumerate(masks_prompts):   
                            
                            sub_mask = [dilate_mask(ma, 15) for ma in mask]
                            img_inpainted_p= f'{save_dir}/mask_{idx}.png'
                            idx=idx+1
                            img_inpainted = models_config['lama'](
                                    im=img_rgb, prompt=sub_mask[0])
                            Image.fromarray(img_inpainted.astype(np.uint8)).save(img_inpainted_p)
                            img_rgb=img_inpainted       
                    for category in categories:
                        if category not in category_colors:
                            category_colors[category] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))            
                    gn = torch.tensor(im.shape)[[1, 0, 1, 0]]  # normalization gain whwh   
                    
                    if (color_flag or save_txt) and(det ) :
                        seg_mask = np.zeros_like(img_rgb)  # img_array 为输入图像的数组表示
                        category_color=[]
                        for xyxy, conf, cls,mask in zip(preds[0],preds[1],preds[2],masks):       #per im boxes              
                                xywh = (xyxy2xywh((xyxy).view(1,4)) / gn).view(-1).tolist()  # normalized xywh   
                                if cls not in categories:
                                    categories.update({
                                            str(cls): len(categories)})        
                                    write_categories(cls,f'{save_dir}/classes_id.txt')
                                    cls_index = len(categories) - 1
                                    category_colors.update({
                                            str(cls):  (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))}) 
                                    category_color=category_colors[str(cls)]
                                else:   
                                    cls_index = categories[str(cls)]
                                    if str(cls) not in category_colors:
                                        category_colors.update({
                                            str(cls):  (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))})
                                    category_color=category_colors[str(cls)]
                                line = (cls_index, xywh, conf) if save_conf else (cls_index, xywh)  # label format
                                line = str(line).replace('[', '').replace(']', '').replace("(",'').replace(")"," ").replace(",", " " * 2)
                                if save_mask:
                                    h, w = mask.shape[-2:]
                                    mask_color = np.array(category_color).reshape((1, 1, -1))  
                                    seg_mask = seg_mask + mask.cpu().numpy().reshape(h, w, 1)  * mask_color  # add    
                                if save_txt:                                 
                                    save_format(label_format="txt",save_path=f'{save_dir}/labels', img_name=name_p, results=line)
                                   
                        if save_mask:
                            plt.figure(figsize=(10,10))
                            plt.imshow(seg_mask)
                            #plt.title('Captioning: ' + caption + '\n' + 'Tagging:' + prompt + '\n')    
                            plt.axis('off')            
                            plt.savefig(os.path.join(f'{save_dir}/masks', f'{name_p}_cls.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)                                                         
                    if save_xml:    
                            h,w=im.shape[:2]
                            save_format("xml",f'{save_dir}/xmls' ,name_p, Path(source).parent, preds, h, w)
                    if det:
                        img_rgb= Image.fromarray(np.uint8(img_rgb), mode='RGB') 
                        draw_img=ImageDraw.Draw(img_rgb) 
                        for box,label in zip(preds[0],preds[2]):   
                            Draw_img( box, draw_img,'box',label,category_colors[str(label)] if color_flag else None)
                    if sam:
                        img_mask=Image.new('RGBA',img_rgb.size,color=(0,0,0,0)  )
                        draw_mask=ImageDraw.Draw(img_mask)       

                        for mask in masks:    
                            Draw_img(mask[0].cpu().numpy(),draw_mask,'mask',None,category_colors[str(label)] if color_flag else None)
                        img_rgb.paste(img_mask, mask=img_mask)  
                    #img_rgb.save(f'{save_dir}/{seen}.jpg')    
                    
            if save_txt:
                #class_ids.append(cls) 
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}/labels")  
            if save_xml:           
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}/xmls")
            if save_caption:
               with open(f'{save_dir}/caption/dataset.json', 'a',encoding='utf-8') as f: 
                    json.dump(JSON_DATASETS,f,ensure_ascii=False) 
                    f.write('\n')
                    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}/captions")
            if save_mask:
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}/masks")
            LOGGER.info('Done...')

            return [[img_rgb],caption,prompt,len(categories)]


def visual_chat(prompt_input, temperature, top_p, image_prompt, result_text,record_audio,upload_audio):
  
    global models_config
    if models_config['visual_glm']:
          if image_prompt and  prompt_input:
              
              return(models_config['visual_glm'].request_model(prompt_input, temperature, top_p, image_prompt, result_text))
             
          else :
               LOGGER.info("请检查你的输入格式和glm模型的参数配置！！！")
    else:                
          return result_text,"没有加载部署的VisualGLM模型!!!"

def clear_fn_image(value):
    return [("", "Hi, What do you want to know ?")]

if __name__ == "__main__":
         
          #check_requirements(exclude=('tensorboard', 'thop'))
          voice_dir='voice_dir'
          if not os.path.exists(voice_dir):
                os.mkdir(voice_dir)
          inputxs=[]
          outputs=[]
          cancel_handles = []
          
          with gr.Blocks(title="Prompt-Can-Anythings",reload=True, theme=adjust_theme(), analytics_enabled=False,full_width=True,css=advanced_css) as block:
               gr.HTML( f"<h1 align=\"center\"> Prompt-Can-Anythings_v1.1 (周更迭代中~)</h1>")
               cookies = gr.State({'api_key': API_KEY, 'llm_model': LLM_MODEL})
               with gr.Row().style(equal_height=False):
                    with gr.Column(scale=1):
                         with gr.Accordion('模型参数和GPU配置', open=False):
                            box_threshold=gr.inputs.Number(label='Confidence Threshold', default=0.3)
                            iou_threshold=gr.inputs.Number(label='Iou Threshold', default=0.5)
                            text_threshold=gr.inputs.Number(label='Text Threshold', default=0.25)
                            device_input=gr.inputs.Textbox(label='device',default='0')
                            quant=gr.inputs.Number(label='quant levels',default=4)                     
                         with gr.Accordion('others Options(标注输出格式设置)', open=False):
                            option_inputs  = {
                            'Save Conf': gr.inputs.Checkbox(label='Save Conf',default=False),
                            'Save img': gr.inputs.Checkbox(label='Save img',default=False),
                            'Visualize': gr.inputs.Checkbox(label='Visualize',default=False),
                            'Project': gr.inputs.Textbox(label='Project:save dir_path',default='runs/detect'),
                            'Name': gr.inputs.Textbox(label='Name',default='exp'),
                            'Exist Ok': gr.inputs.Checkbox(label='Exist Ok',default=False)
                            }   
                           
                         inputxs.extend(list(option_inputs.values()))
                         with gr.Accordion('Method_Options:free combo', open=True):                
                                   
                                methods_options={'Lama': gr.inputs.Checkbox(label='Lama model[近期更新测试中]',default=False), 
                                                'Sam': gr.inputs.Checkbox(label='Sam[当前仅支持检测器的BOX输入]',default=False),
                                                'Det': gr.inputs.Checkbox(label='Grounded[可输入文本的检测器]',default=False), 
                                                'Tag2text': gr.inputs.Checkbox(label='Tag2text[图文理解]',default=False)
                                }
                                
                                visual_glm=gr.inputs.Checkbox(label='VisualGLM',default=False)
                                chatgpt=gr.inputs.Checkbox(label='ChatGPT(目前为网络服务自动挂载)',default=True)
                                
                                loads_model_button=gr.Button('热重载模型',variant="primary")
                                loads_flag=gr.inputs.Textbox(label="加载模型进度")
                            
                         list_methods=list(methods_options.values())
                         inputxs.extend(list_methods)  
                       
                         with gr.Accordion('format Options', open=False):                
                                   
                                save_options={
                                'Save txt': gr.inputs.Checkbox(label='Save txt [collect class nums]',default=False), 
                                'Save xml': gr.inputs.Checkbox(label='Save xml',default=False), 
                                'Save Mask': gr.inputs.Checkbox(label='Save Mask',default=False),  
                                'Save Caption': gr.inputs.Checkbox(label='Save Caption',default=False), 
                                'Batch Process': gr.inputs.Checkbox(label='Batch Process[暂不支持]',default=False), 
                                'Color Flag': gr.inputs.Checkbox(label='Color Flag:must check[Save txt]',default=False)
                            }
                         inputxs.extend(list(save_options.values()))
                         dir_inputs =gr.inputs.Textbox(label='加载本地图像文件夹路径',default='train_imgs')
                         with gr.Accordion('LLM模型配置', open=False):
                            md_dropdown = gr.Dropdown(AVAIL_LLM_MODELS, value=LLM_MODEL, label="更换LLM模型/请求源 [暂时仅支持chatgpt]").style(container=False)
                            max_length_sl = gr.Slider(minimum=256, maximum=4096, value=512, step=1, interactive=True, label="Local LLM MaxLength")
                            with gr.Row():
                                top_p = gr.Slider(minimum=-0, maximum=1.0, value=1.0, step=0.01,interactive=True, label="nucleus sampling",)
                                temperature = gr.Slider(minimum=-0, maximum=2.0, value=1.0, step=0.01, interactive=True, label="Temperature",)
                         with gr.Accordion('VisualGLM模型配置', open=False):
                              visual_temperature = gr.Slider(maximum=1, value=0.8, minimum=0, label='VisualGLMTemperature')
                              visual_top_p = gr.Slider(maximum=1, value=0.4, minimum=0, label='VisualGLM Top_P')
                         
                         with gr.Accordion('语音服务模型配置', open=True):
                                            with gr.Row():
                                                asr_select = gr.inputs.Checkbox(label='use ASR',default=False).style(height=10,width=10)
                                                asr_button = gr.Button('loads ASR').style(height=10,width=10)     
                    with gr.Column(variant='panel',scale=15):      
                         with gr.Row():
                                    with gr.Row():
                                        record_audio = gr.Audio(label="record your voice", source="microphone",type='filepath').style(width=120)
                                        send_record_button = gr.Button('声音转文本到输入框里同步[修复中]')
                                     
                                              
                         with gr.Tabs(elem_id="Process_audio"):
                                with gr.TabItem('Upload OR TTS[近期更新—未连GPT]'):
                                        
                                        with gr.Column(variant='panel'):
                                            with gr.Row():
                                                upload_audio = gr.Audio(label="Input audio(./wav/.mp3)", source="upload",type='filepath').style(height=60,width=120)
                                                input_text = gr.Textbox(label="Generating audio from text", lines=2, placeholder="please enter some text here, we genreate the audio from  TTS.")
                                            with gr.Row():
                                                asr = gr.Button('Generate text[时间太长的内容可能前端不稳定]',elem_id="text_generate", variant='primary')
                                                tts = gr.Button('Generate audio',elem_id="audio_generate", variant='primary')   
                                                #with gr.Row():
                                                audio_chat=gr.Button('send_chat["X"暂时不支持]', variant='primary')   
                                              
                                                              
                        #  if sys.platform!='win32':
                        #             from utils.text2speech import T2S
                        #             tts_model=T2S()
                        #             t2s=tst.test   
                         import edge_tts
                         def t2s(text):
                                    asyncio.run(t2s_inference(text))
                                    return voice_dir+"/temp.wav"
                         async def t2s_inference(text):
                                    generate_wave = edge_tts.Communicate(text, voice='zh-CN-YunxiNeural', rate='-5%', volume='+1%')
                                    await generate_wave.save(voice_dir+"/temp.wav")   
                         def s2t(speech_file,stream_mode=False):
                            #global speech_AI
                            from a2f import speech_recognition
                            speech_text, speech_language=speech_recognition(speech_file, speech_AI['whisper'],stream_mode)                             #
                            return  speech_text  
                        
                         with gr.Tabs(elem_id="上传图像"):
                                with gr.TabItem('Upload image'):
                                        with gr.Row():
                                            image_prompt = gr.Image(label="Source image", source="upload", type="filepath").style(height=200,width=180)
                                      
                         prompt_input=gr.inputs.Textbox(lines=2, label="prompt with image/仅与图像相关提示词 : (Optional,注意在使用每一个功能前请考虑在这个框里的TEXT提示词要不要先清空)")
                         run_button = gr.Button('Run CV_Task',variant="primary")
                       
                         inputs = [dir_inputs,image_prompt,prompt_input,box_threshold,iou_threshold,text_threshold,device_input,quant]
                         inputs.extend(inputxs)

                         with gr.Row():
                                run_button_2 = gr.Button('VisualGLM',variant="primary")
                                clear_button= gr.Button("清除", variant="secondary")
                                status = gr.Markdown(f"Tip: 按Enter提交, 按Shift+Enter换行。当前模型: {LLM_MODEL} \n ")
                         with gr.Row():

                                resetBtn = gr.Button("重置", variant="secondary"); resetBtn.style(size="sm")
                                stopBtn2 = gr.Button("停止", variant="secondary"); stopBtn2.style(size="sm")
                             
                         chat_txt=gr.Textbox(lines=3,show_label=False, placeholder="question").style(container=False)
                         with gr.Accordion("备选输入区", open=True, visible=False) as area_input_secondary:
                            with gr.Row():
                                txt = gr.Textbox(show_label=False, placeholder="Input question here.", label="输入区2").style(container=False)
                         run_button_chat = gr.Button('Chat_Sumbit',variant="primary")
                         with gr.Accordion("学术ChatGPT基础功能", open=True) as area_basic_fn:
                              with gr.Row():
                                for k in functional:
                                    if ("Visible" in functional[k]) and (not functional[k]["Visible"]): continue
                                    variant = functional[k]["Color"] if "Color" in functional[k] else "secondary"
                                    functional[k]["Button"] = gr.Button(k, variant=variant)   
                       
                         with gr.Row():
                                plugin_advanced_arg = gr.Textbox(show_label=True, label="高级参数输入区", visible=False, 
                                                                 placeholder="这里是特殊函数插件的高级参数输入区").style(container=False)

                         video_output = gr.Video(type="auto")                                   
                    with gr.Column(scale=20):
                     
                         gallery = gr.Gallery(label="Generated images",show_label=False,elem_id="gallery",).style(preview=True, grid=2, object_fit="scale-down")
                         with gr.Row():
                            output_text = gr.Textbox(label="图像理解",lines=2)
                            zh_select=gr.inputs.Checkbox(label='翻译【勾选后需要重新一键重载模型】',default=False)
                            with gr.Column():
                                output_classes= gr.Textbox(label="Class_numbers:auto generate classes numbers, color flag or save_txt must be ture ")
                                output_tag= gr.outputs.Textbox(label="Tag")
                         with gr.Row():
                            with gr.Accordion("备选输入区", open=True, visible=False) as area_input_secondary:
                                 system_prompt = gr.Textbox(show_label=True, placeholder=f"Chat Prompt", label="下方输入对话支持图像和文本", value="AI assistant.")
                                            
                         with gr.Row():
                            with gr.Column(scale=2):
                                result_text = gr.Chatbot(label=f'Multi-round conversation History,当前模型：{LLM_MODEL}', value=[("", "Hi, What do you want to know ?")]).style(height=CHATBOT_HEIGHT)
                                history = gr.State([])
                                
               send_record_button.click(fn=s2t, inputs=[record_audio], outputs=[input_text])                       
               asr_button.click(fn=load_speech_model,inputs=[asr_select],outputs=[loads_flag])        
               asr.click(fn=s2t, inputs=[upload_audio], outputs=[input_text])                    
               tts.click(fn=t2s, inputs=[input_text], outputs=[upload_audio])        
                        
               cs=[]                 
               cs.extend(list_methods)  
               cs.extend([zh_select, visual_glm,device_input, quant, loads_flag])
               loads_model_button.click(fn=load_auto_backend_models,inputs=cs,outputs=[loads_flag])                     
               inputs.append(zh_select)
               outputs = [gallery, output_text, output_tag,output_classes]             
               input_combo = [cookies, max_length_sl,md_dropdown,chat_txt,txt,top_p, temperature, result_text, history,system_prompt,plugin_advanced_arg]   
               # = [cookies, max_length_sl,md_dropdown,input_text,txt,top_p, temperature, result_text, history,system_prompt,plugin_advanced_arg]      
               output_combo = [cookies, result_text, history, status]
               predict_args = dict(fn=ArgsGeneralWrapper(predict), inputs=input_combo, outputs=output_combo)  
               #predict_args = dict(fn=ArgsGeneralWrapper(predict), inputs=input_combo, outputs=output_combo)  
               run_button.click(fn=Auto_run, inputs=inputs, outputs=outputs)
                # 提交按钮、重置按钮
               cancel_handles.append(chat_txt.submit(**predict_args))
               cancel_handles.append(txt.submit(**predict_args))
               cancel_handles.append(run_button_chat.click(**predict_args))
               cancel_handles.append(clear_button.click(**predict_args))
               resetBtn.click(lambda: ([], [], "已重置"), None, [result_text, history, status])
               stopBtn2.click(fn=None, inputs=None, outputs=None, cancels=cancel_handles)
              
               for k in functional:
                    if ("Visible" in functional[k]) and (not functional[k]["Visible"]): continue
                    dict_args=dict(fn=ArgsGeneralWrapper(predict), inputs=[*input_combo, gr.State(True),gr.State(k)], outputs=output_combo)
                    
                    cancel_handles.append(functional[k]["Button"].click(**dict_args))

               def on_md_dropdown_changed(k):
                    return {result_text: gr.update(label="当前模型："+k)}
               md_dropdown.select(on_md_dropdown_changed, [md_dropdown], [result_text])
               
               #VisualGLM         
               run_button_2.click(fn=visual_chat,inputs=[prompt_input, visual_temperature, visual_top_p, image_prompt,
                                                         result_text,record_audio,upload_audio],
                                outputs=[prompt_input, result_text])
               prompt_input.submit(fn=visual_chat,inputs=[prompt_input, visual_temperature, visual_top_p, image_prompt,
                                                         result_text,record_audio,upload_audio],
                                        outputs=[prompt_input, result_text])
               #clear_button.click(fn=clear_fn, inputs=clear_button, outputs=[prompt_input, result_text, image_prompt])
               image_prompt.upload(fn=clear_fn_image, inputs=clear_button, outputs=[result_text])
               clear_button.click(lambda: ("","","","",""), None, [prompt_input,result_text,txt, input_text,chat_txt])
               image_prompt.clear(fn=clear_fn_image, inputs=clear_button, outputs=[result_text])
              # upload_audio.clear(fn=clear_fn_image, inputs=clear_button, outputs=[result_text])
          auto_opentab_delay(7589)
          block.queue(concurrency_count=CONCURRENT_COUNT).launch(server_name='0.0.0.0', server_port=7589,debug=True, share=False)
     

     
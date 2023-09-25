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
from llm_cards.bridge_all import predict_all,talk_all
from llm_cards.bridge_chatgpt import Talk_with_app
from llm_cards.core_functional import get_core_functions
from utils.toolbox import format_io, find_free_port, on_file_uploaded, on_report_generated, get_conf, ArgsGeneralWrapper, load_chat_cookies, DummyWith

from utils.torch_utils import select_device
from utils import VID_FORMATS,IMG_FORMATS,write_categories

import gradio as gr
import random
import json
import multiprocessing as mp
import asyncio
import concurrent.futures
from utils.colorful import *

functional = get_core_functions()

VisualGLM_dir=f"VisualGLM_6B"
sys.path.append(VisualGLM_dir)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  #  root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

global categories
categories = {}
global category_colors
category_colors={}
# 初始对应类别编号
class_ids = []
global speech_AI
speech_AI={'asr':{'whisper':None},'tts':{'tts_VITS':None,'tts_edge': None}} ## speech 
global models_config
models_config = {'tag2text': None, 'ram': None,'lama': None,'sam': None,'grounded': None,'sd': None,    ## cv with text
                 'visual_glm': None , 'trans_zh': None,'gligen': None}
NUM_WORKERS=1
JSON_DATASETS=[]


operation_running = False

def toggle_operation(flag):

        import whisper
        from a2f import speech_recognition,mic_audio,keyboard
        if speech_AI['asr']['whisper'] is  None:
            speech_AI['asr']['whisper']=whisper.load_model("small",
                                download_root="weights")
        
        print("asr加载完毕,开始录音!")
        text=[]
        speech_text=''
        while True:
               # result_txt="你好我没有正确识别到结果"
                if keyboard.is_pressed('q'):  
                    mic_audio('voice_dir/send_asr.wav')
                    speech_text,__=speech_recognition('voice_dir/send_asr.wav',speech_AI['asr']['whisper'],False) 
                    break
        print(speech_text) 
        text.append(speech_text)
        return  text

async def sadtalker_demo(checkpoint_path,config_path,source_image,
                            driven_audio,
                            preprocess_type,
                            is_still_mode,
                            enhancer,
                            batch_size,                            
                            size_of_image,
                            pose_style,
                            exp_weight):
        sys.path.append('SadTalker')
        from SadTalker.app import SadTalker
    
        sadtaker_model=SadTalker(checkpoint_path, config_path, lazy_load=True)
        output = await asyncio.to_thread(sadtaker_model.test, source_image,
                            driven_audio,
                            preprocess_type,
                            is_still_mode,
                            enhancer,
                            batch_size,                            
                            size_of_image,
                            pose_style,
                            exp_weight)
        return output
         
def train_visualGLM(name,model_size,mode,train_iters,resume_data,
        max_source_length,max_target_length,lora_rank,layer_range_s,layer_range_e,pre_seq_len,
       train_data,valid_data,distributed_backend,lr_decay_style,warmup,
       checkpoint_activations,save_interval,eval_interval,save_path,
       split,eval_iters,eval_batch_size ,zero_stage,
       lr,batch_size,accumulation_steps,method_type):
    
    model_args=[max_source_length,max_target_length,lora_rank,layer_range_s,layer_range_e,pre_seq_len]
    gpt_option=[name,int(model_size),mode,int(train_iters),resume_data, #23 
       train_data,valid_data,distributed_backend,lr_decay_style,warmup, 
       checkpoint_activations,int(save_interval),int(eval_interval),save_path,
       int(split),int(eval_iters),int(eval_batch_size),int(zero_stage),
       lr,int(batch_size),int(accumulation_steps)]
    
    processes = []
    for i in range(NUM_WORKERS):
         p = mp.Process(target=start_finetuning_process, args=(gpt_option,model_args,method_type))
         p.start()
         processes.append(p)
    for p in processes:
        p.join()
    return 'OK'    

#具体参数待修复调整
def start_finetuning_process(gpt_option,model_args,method_type):
    print('fine subprocess start')
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    print(script_dir+'/'+VisualGLM_dir)
    main_dir = os.path.dirname(script_dir)
  
    model_args = f'--max_source_length {model_args[0]} --max_target_length {model_args[1]} --lora_rank {model_args[2]} --layer_range {model_args[3]}  {model_args[4]} --pre_seq_len {model_args[5]}'
    options_nccl = 'NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2'
    host_file_path = 'hostfile_single'
   
    gpt_option_prefix=f" \
       --experiment-name finetune-{gpt_option[0]} \
       --model-parallel-size {gpt_option[1]} \
       --mode {gpt_option[2]} \
       --train-iters {gpt_option[3]} \
       --resume-dataloader \
        {model_args} \
       --train-data {gpt_option[5]} \
       --valid-data {gpt_option[6]} \
       --distributed-backend {gpt_option[7]} \
       --lr-decay-style {gpt_option[8]}\
       --warmup {gpt_option[9]} \
       --checkpoint-activations \
       --save-interval {gpt_option[11]} \
       --eval-interval {gpt_option[12]} \
       --save {gpt_option[13]} \
       --split {gpt_option[14]}\
       --eval-iters {gpt_option[15]} \
       --eval-batch-size {gpt_option[16]}\
       --zero-stage {gpt_option[17]} \
       --lr {gpt_option[18]} \
       --batch-size {gpt_option[19]} "
    lora=f"  \
       --skip-init  \
       --fp16  \
       --use_lora "
    qlora=f"--gradient-accumulation-steps {gpt_option[20]} \
       --skip-init \
       --fp16 \
       --use_qlora"
    ptune=f"  \
       --skip-init  \
       --fp16  \
       --use_ptuning"  
    if method_type=='use_qlora':
        gpt_options=gpt_option_prefix+qlora
    elif method_type=='use_lora':
        gpt_options=gpt_option_prefix+lora
    elif method_type=='use_ptuning':
        gpt_options=gpt_option_prefix+ptune
    else:    
        LOGGER.info("没有选择训练方法！！！")   
        return   
      
    run_cmd = f'{options_nccl} deepspeed --master_port 16666 --hostfile {host_file_path} {VisualGLM_dir}/finetune_visualglm.py {gpt_options} '
    os.system(run_cmd)

async def load_speech_model(asr_method,tts_method):
        import whisper 
        global speech_AI
        if asr_method=='whisper' :
            speech_AI['asr']['whisper']=  whisper.load_model("small",download_root="weights")
            LOGGER.info('loads whisper')
            
        elif not asr_method and speech_AI['asr']['whisper']:
            LOGGER.info('free  memory')
            speech_AI['asr']['whisper']=None  
        else:    
            LOGGER.info('pass')  
          
        if tts_method =="VITS":
             print('调试中，很快更新')
        #          speech_AI['tts']['VITS'] =  
        #    LOGGER.info('loads whisper')
            
        elif not tts_method:
            LOGGER.info('pass')
        return '语音识别记载完成'

               
def save_text2img_data(prompt,label,img_name,zh_select):
    global JSON_DATASETS
    if not prompt :
        prompt=f"这张图片的背景里有什么内容?"
    if not zh_select:
        prompt=f'What contents are present in the background of this picture?'
    example = {
        "img": f"{img_name}",
        "prompt": prompt,
        "label": label
    }
    JSON_DATASETS.append(example)
            
async def load_auto_backend_models(lama, sam, det,tag2text,ram, trans_zh, visual_glm,device=0, quant=4, bar=None): 
    try:    
        with concurrent.futures.ThreadPoolExecutor() as pool:
                wait_coros =  asyncio.get_event_loop().run_in_executor(pool, load_auto_backend_model, lama, sam, det, tag2text,ram,trans_zh, visual_glm,device, quant, bar)
                await asyncio.wait([wait_coros])
        await asyncio.sleep(0.01) 
    except Exception as e:
        LOGGER.info("An error occurred: ", e)
        return 'windows可能会出现问题,请再次点击加载按钮，也可以检查后台'
    return 'Loads Done !'
   

def load_auto_backend_model(lama,sam,det,tag2text,ram,trans_zh,visual_glm,device,quant,bar):
    """
    加载模型库
    """
    # Load model    
    
    global models_config
 
    if visual_glm and not models_config['visual_glm']:
          from VisualGLM_6B.chatglm import  VisualGLM
          models_config['visual_glm']=VisualGLM(gpu_device=int(device),quant=int(quant))
          LOGGER.info(f'GPU{int(device)}———量化VisualGLM模型:int{int(quant)}')
    elif not visual_glm:
          LOGGER.info('no select visualGLM')
          models_config['visual_glm']=None  
    else:    
          LOGGER.info('free or no visual_glm')      
           
    device = select_device(device)    
    if tag2text and not models_config['tag2text']:
                models_config['tag2text'] = AutoBackend("tag2text",weights=Tag2Text_Model_Path,device=device)              
    elif not tag2text  :
            LOGGER.info('no tag2text')
            models_config['tag2text'] =None 
    else :
            LOGGER.info('free or tag2text pass')   
             
    if det and not models_config['grounded']:
            models_config['grounded'] = AutoBackend("grounded-DINO",weights=GROUNED_MODEL_TYPE['S'], device=device,
            args_config= 'model_cards/groundingdino/config/GroundingDINO_SwinT_OGC.py')
    elif not det  :
            models_config['grounded'] =None 
    else :
            LOGGER.info('free or grounded pass')
            
    if sam and not models_config['sam']:
            models_config['sam']= AutoBackend("segment-anything",weights=SAM_MODEL_TYPE['vit_h'] ,device=device)
    elif not sam :
            models_config['sam'] =None      
    else:
            LOGGER.info("PASS SAM")

    if ram and not models_config['ram']:
            LOGGER.info("ram loads")
            models_config['ram']= AutoBackend('ram',weights=Ram_Model_Path ,device=device)
    elif not ram :
            models_config['ram'] =None      
    else:
            LOGGER.info("PASS ram")       

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
        tag2text=False,
        ram=False,
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
                            save_text2img_data(None, caption,name_p,zh_select)
                            #save_format(label_format="txt",save_path=f'{save_dir}/captions',img_name=name_p, results=caption)
                    if ram:
                        LOGGER.info(f'ram No need prompt:{prompt}')
                        en_tag,zh_tag = models_config['ram'](im = img_rgb,prompt=prompt,box_threshold=conf_thres,text_threshold=text_thres,iou_threshold=iou_thres)
                       
                        prompt=en_tag.replace(' |', ',')
                        zh_tag=zh_tag.replace(' |', ', ')
                        #LOGGER.info(preds[1])
                        LOGGER.info(f"en_Tags: {prompt}")
                        print(f"zh_Tags : {zh_tag}")
                        # if zh_select and prompt :
                        #     caption=models_config['trans_zh'](caption, max_length=1000, clean_up_tokenization_spaces=True)[0]["generated_text"]
                        # if save_caption:
                        #     save_text2img_data(None, caption,name_p,zh_select)

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
               with open(f'{save_dir}/captions/dataset.json', 'a',encoding='utf-8') as f: 
                    json.dump(JSON_DATASETS,f,ensure_ascii=False) 
                    f.write('\n')
                    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}/captions")
            if save_mask:
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}/masks")
            LOGGER.info('Done...')

            return [[img_rgb],caption,prompt,len(categories)]


def visual_chat(prompt_input, temperature, top_p, image_prompt, result_text,record_audio,upload_audio,omniverse=False):
  
    global models_config
    print(f"是否连接omniverse:{omniverse}")
    if models_config['visual_glm']:
          if image_prompt and  prompt_input:

                __, result_text=(models_config['visual_glm'].request_model(prompt_input, temperature, top_p, image_prompt, result_text))
                if omniverse: 
                        from a2f import tts_a2f       
                        asyncio.run(tts_a2f(result_text[-1][-1]))
                return "",result_text
          else :
               LOGGER.info("请检查你的输入格式和glm模型的参数配置！！！")
    else:                
          return result_text,"没有加载部署的VisualGLM模型!!!"

def clear_fn_image(value):
    return [("", "Hi, What do you want to know ?或者你想从图像中知道什么?")]

if __name__ == "__main__":
         
          #check_requirements(exclude=('tensorboard', 'thop'))
          proxies, WEB_PORT, LLM_MODEL, CONCURRENT_COUNT, AUTHENTICATION, CHATBOT_HEIGHT, LAYOUT, AVAIL_LLM_MODELS, AUTO_CLEAR_TXT = \
          get_conf('proxies', 'WEB_PORT', 'LLM_MODEL', 'CONCURRENT_COUNT', 'AUTHENTICATION', 'CHATBOT_HEIGHT', 'LAYOUT', 'AVAIL_LLM_MODELS', 'AUTO_CLEAR_TXT')
          AUTO_CLEAR_TXT = get_conf('AUTO_CLEAR_TXT')
        # 如果WEB_PORT是-1, 则随机选取WEB端口
          PORT = find_free_port() if WEB_PORT <= 0 else WEB_PORT
          functional = get_core_functions()
          from themes.theme import adjust_theme, advanced_css, theme_declaration
            # 高级函数插件
          from llm_cards.crazy_functional import get_crazy_functions
          crazy_fns = get_crazy_functions()
          import logging, uuid
          os.makedirs("gpt_log", exist_ok=True)
          try:logging.basicConfig(filename="gpt_log/chat_secrets.log", level=logging.INFO, encoding="utf-8", format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
          except:logging.basicConfig(filename="gpt_log/chat_secrets.log", level=logging.INFO,  format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        # Disable logging output from the 'httpx' logger
          logging.getLogger("httpx").setLevel(logging.WARNING)
          print("所有问询记录将自动保存在本地目录./gpt_log/chat_secrets.log, 请注意自我隐私保护哦！")
        # 处理markdown文本格式的转变
          gr.Chatbot.postprocess = format_io
          # 代理与自动更新
          from utils.check_proxy import check_proxy, auto_update, warm_up_modules
          proxy_info = check_proxy(proxies)
          voice_dir='voice_dir'
          if not os.path.exists(voice_dir):
                os.mkdir(voice_dir)
          inputxs=[]
          outputs=[]
          cancel_handles = []
          
          with gr.Blocks(title="Prompt-Can-Anythings",reload=True, theme=adjust_theme(), analytics_enabled=False,full_width=True,css=advanced_css) as block:
               gr.HTML( f"<h1 align=\"center\"> Prompt-Can-Anythings_v1.15 (周更迭代中)</h1>")
               cookies = gr.State({'api_key': API_KEY, 'llm_model': LLM_MODEL})
               with gr.Row().style(equal_height=False):
                    with gr.Column(scale=1):
                         with gr.Accordion('视觉模型配置', open=False):             
                            with gr.TabItem('本地模型配置'):
                                    box_threshold=gr.inputs.Number(label='Confidence Threshold', default=0.3)
                                    iou_threshold=gr.inputs.Number(label='Iou Threshold', default=0.5)
                                    text_threshold=gr.inputs.Number(label='Text Threshold', default=0.25)
                                    device_input=gr.inputs.Textbox(label='device',default='0')
                                    quant=gr.inputs.Number(label='quant levels',default=4)   
                                     
                            with gr.TabItem('其他【不需要修改】'):
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
                                                'Tag2text': gr.inputs.Checkbox(label='Tag2text[图文理解]',default=False),
                                                'ram': gr.inputs.Checkbox(label='ram[识别标签]',default=False)
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
                                'Color Flag': gr.inputs.Checkbox(label='Color Flag[标识语义]',default=False)
                            }
                         inputxs.extend(list(save_options.values()))
                         dir_inputs =gr.inputs.Textbox(label='加载本地图像文件夹路径',default='train_imgs')
                         with gr.Accordion('LLM模型配置', open=False):
                            checkboxes = gr.CheckboxGroup(["基础功能区", "函数插件区", "底部输入区", "输入清除键", "插件参数区"], value=["基础功能区", "函数插件区"], label="显示/隐藏功能区")
                            md_dropdown = gr.Dropdown(AVAIL_LLM_MODELS, value=LLM_MODEL, label="更换LLM模型源 [暂时仅支持chatgpt/glm2]").style(container=False)
                            max_length_sl = gr.Slider(minimum=256, maximum=4096, value=512, step=1, interactive=True, label="Local LLM MaxLength")
                            with gr.Row():
                                quant_chatglm= gr.Dropdown(MODEL_QUANTIZE,value=None,label="llm quantize[chatglm] ").style(container=False)
                                top_p = gr.Slider(minimum=-0, maximum=1.0, value=1.0, step=0.01,interactive=True, label="nucleus sampling",)
                                temperature = gr.Slider(minimum=-0, maximum=2.0, value=1.0, step=0.01, interactive=True, label="Temperature",)
                         with gr.Accordion('VisualGLM模型配置', open=False):
                              visual_temperature = gr.Slider(maximum=1, value=0.8, minimum=0, label='VisualGLMTemperature')
                              visual_top_p = gr.Slider(maximum=1, value=0.4, minimum=0, label='VisualGLM top_P')
                         
                         with gr.Accordion('语音模型配置', open=False):
                                            with gr.Row():
                                                asr_select = gr.Dropdown(ASR_METHOD,value='whisper', label="语音识别方法").style(container=False)
                                                tts_select = gr.Dropdown(TTS_METHOD,value='VITS', label="语音合成方法").style(container=False)
                                                asr_gpt = gr.inputs.Checkbox(label='ASR gpt [无需加载按钮]',default=False).style(height=1,width=1)
                                                asr_button = gr.Button('Loads SPEECH_AI').style(height=5,width=5)  
                         with gr.Accordion('大模型对话系统配置', open=True):
                                            with gr.Row():                                        
                                                chat_app = gr.inputs.Checkbox(label='start system',default=False).style(height=1,width=1)
                                                chat_app_button = gr.Button('Speech_system').style(height=5,width=5)  
                         with gr.Accordion('ViusalGLM训练配置', open=False):
                                with gr.Row():
                                    train_methods=gr.Dropdown(AVAIL_METHOD_FINETUNE,value=METHOD_FINETUNE, label="微调方法").style(container=False)
                                    visualglm_args=[                              
                                    gr.inputs.Textbox(label="Experiment_Name", default="visualglm-6b"),
                                    gr.inputs.Number(label="Model Parallel Size", default=1),
                                    gr.inputs.Textbox(label="mode", default='finetune'),
                                    gr.Slider(minimum=1, maximum=3000, value=300, step=1, interactive=True, label="train-iters"),
                                    gr.inputs.Checkbox(label="resume dataloader", default=True),
                                    gr.Slider(minimum=16, maximum=256, value=64, step=1, interactive=True, label="max_source_length"),
                                    gr.Slider(minimum=16, maximum=1024, value=256, step=1, interactive=True, label="max_target_length"),
                                    gr.Slider(minimum=1, maximum=100, value=10, step=1, interactive=True, label="lora_rank"),
                                    gr.Slider(minimum=0, maximum=256, value=0, step=1, interactive=True, label="layer_range_start"),
                                    gr.Slider(minimum=0, maximum=20, value=14, step=1, interactive=True, label="layer_range_end"),
                                    gr.Slider(minimum=1, maximum=60, value=4, step=1, interactive=True, label="pre_seq_len"),          
                                    gr.inputs.Textbox(label="Train Data", default="fewshot-data/dataset.json"),
                                    gr.inputs.Textbox(label="Eval Data", default="fewshot-data/dataset.json"),
                                    gr.inputs.Textbox(label="distributed backend", default="nccl"),
                                    gr.inputs.Dropdown(label="lr decay style ", choices=["cosine", "linear"], default="cosine"),
                                    gr.inputs.Number(label="warmup", default=0.02),
                                    gr.inputs.Checkbox(label="checkpoint-activations", default=True) ,
                                    gr.inputs.Number(label="Save Interval", default=300),
                                    gr.inputs.Number(label="Eval Interval", default=10000),
                                    gr.inputs.Textbox(label="Save Directory", default="./checkpoints"),
                                    gr.inputs.Number(label="split", default=1),
                                    gr.inputs.Number(label="Eval Iters", default=10),
                                    gr.inputs.Number(label="Eval Batch Size", default=8),
                                    gr.inputs.Textbox(label='Zero Stage',default=1),
                                    gr.inputs.Number(label="lr", default=0.0001),
                                    gr.inputs.Number(label="batch size", default=4),
                                    gr.inputs.Number(label="gradient accumulation steps", default=4),
                                    ]
                         fine_tune=gr.Button('Finetune VisualGLM').style(height=5,width=5)   
                                    
                         with gr.Accordion('sadtakler配置', open=False):
                            with gr.Tabs(elem_id="sadtalker_checkbox"):
                                with gr.TabItem('Settings'):
                                    gr.Markdown("need help? please visit our [[best practice page](https://github.com/OpenTalker/SadTalker/blob/main/docs/best_practice.md)] for more detials")
                                    with gr.Column(variant='panel'):
                                            # width = gr.Slider(minimum=64, elem_id="img2img_width", maximum=2048, step=8, label="Manually Crop Width", value=512) # img2img_width
                                            # height = gr.Slider(minimum=64, elem_id="img2img_height", maximum=2048, step=8, label="Manually Crop Height", value=512) # img2img_width
                                            with gr.Row():
                                                pose_style = gr.Slider(minimum=0, maximum=46, step=1, label="Pose style", value=0) #
                                                exp_weight = gr.Slider(minimum=0, maximum=3, step=0.1, label="expression scale", value=1) # 
                                            with gr.Row():
                                                sadtalker_path=gr.inputs.Textbox(label="checkpoint path", default="checkpoints") 
                                                sadtalker_config=gr.inputs.Textbox(label="config path", default="SadTalker/src/config")
                                            with gr.Row():
                                                size_of_image = gr.Radio([256, 512], value=256, label='face model resolution', info="use 256/512 model?") # 
                                                preprocess_type = gr.Radio(['crop', 'resize','full', 'extcrop', 'extfull'], value='crop', label='preprocess', info="How to handle input image?")
                                            
                                            with gr.Row():
                                                is_still_mode = gr.Checkbox(label="Still Mode (fewer hand motion, works with preprocess `full`)")
                                                batch_size = gr.Slider(label="batch size in generation", step=1, maximum=10, value=2)
                                                enhancer = gr.Checkbox(label="GFPGAN as Face enhancer")
                                               
                                            sadtalker_submit = gr.Button('Generate_video', elem_id="sadtalker_generate", variant='primary')
                                       
                    with gr.Column(variant='panel',scale=15):      
                       
                         with gr.Tabs(elem_id="Process_audio"):
                                with gr.TabItem('Upload OR TTS'):
                                        
                                        with gr.Column(variant='panel'):
                                            with gr.Row():
                                                record_audio = gr.Audio(label="record your voice", source="microphone",type='filepath')
                                                #Recording_audio=gr.Button('Recording_asr',elem_id="speech2text", variant='primary')
                                                
                                            with gr.Row():
                                                upload_audio = gr.Audio(label="Input audio(./wav/.mp3)", source="upload",type='filepath').style(height=20,width=120)
                                                input_text = gr.Textbox(label="Generating audio from text", lines=2, placeholder="please enter some text here, we genreate the audio from  TTS.")
                                               
                                            with gr.Row():
                                                asr = gr.Button('Generate text',elem_id="text_generate", variant='primary')
                                                tts = gr.Button('Generate audio',elem_id="audio_generate", variant='primary')   
                                with gr.TabItem('Omniverse App'):   
                                        with gr.Row():
                                            omniverse_switch = gr.inputs.Checkbox(label='Omniverse A2F',default=False)
                                            #audio_to_face=gr.Button('send a Audio to Omniverse ', variant='primary')  
                                               
                         def t2s(text,method):
                                    from a2f import tts_send2
                                    send_dir=f'{voice_dir}/send_a2f.wav'
                                    if method=='VITS':
                                        print('更新中，暂不支持')
                                    elif method=='edge_tts'  :  
                                        asyncio.run(tts_send2(text,False,send_dir))
                                    return send_dir
                                                    
                         def s2t(speech_file,stream_mode=False):
                            from a2f import speech_recognition
                            speech_text, speech_language=speech_recognition(speech_file, speech_AI['asr']['whisper'],stream_mode)                             #
                            return  speech_text  
                        
                         with gr.Tabs(elem_id="上传图像"):
                                with gr.TabItem('Upload image'):
                                        with gr.Row():
                                            image_prompt = gr.Image(label="Source image", source="upload", type="filepath").style(height=200,width=180)
                                      
                         prompt_input=gr.inputs.Textbox(lines=2, label="prompt with image/仅与图像相关 : (Optional,注意每个功能请考虑在这个框里的TEXT提示词要不要先清空)")
                           
                         inputs = [dir_inputs,image_prompt,prompt_input,box_threshold,iou_threshold,text_threshold,device_input,quant]
                         inputs.extend(inputxs)

                         with gr.Row():
                                run_button = gr.Button('Run CV_Task',variant="primary"); run_button.style(size="sm")
                                clear_button= gr.Button("清除文本", variant="secondary"); clear_button.style(size="sm")
                        
                         with gr.Row():
                                    resetBtn = gr.Button("重置", variant="secondary"); resetBtn.style(size="sm")
                                    stopBtn2 = gr.Button("停止", variant="secondary"); stopBtn2.style(size="sm")
                                    clearBtn = gr.Button("清除", variant="secondary", visible=False); clearBtn.style(size="sm")
                         with gr.Row():       
                                status = gr.Markdown(f"Tip: 按Enter提交, 按Shift+Enter换行。当前模型: {LLM_MODEL} \n {proxy_info}", elem_id="state-panel")        
                         with gr.Tabs(elem_id="Chatbox"): 
                            with gr.TabItem('对话区'):  
                                with gr.Accordion("输入区", open=True, elem_id="input-panel") as area_input_primary: 
                                    with gr.Row():  
                                        chat_txt=gr.Textbox(lines=3,show_label=False, placeholder="question").style(container=False)
                         with gr.Accordion("备选输入区", open=True, visible=False) as area_input_secondary:
                            with gr.Row():
                                txt = gr.Textbox(show_label=False, placeholder="Input question here.", label="输入区2").style(container=False)
                         with gr.Row():
                            run_button_chat = gr.Button('Chat_Sumbit',variant="primary")
                            run_button_2 = gr.Button('VisualGLM',variant="primary")
                         with gr.Accordion("学术ChatGPT基础功能", open=False) as area_basic_fn:
                              with gr.Row():
                                for k in functional:
                                    if ("Visible" in functional[k]) and (not functional[k]["Visible"]): continue
                                    variant = functional[k]["Color"] if "Color" in functional[k] else "secondary"
                                    functional[k]["Button"] = gr.Button(k, variant=variant)   
                         with gr.Accordion("函数插件区", open=False, elem_id="plugin-panel") as area_crazy_fn:
                            with gr.Row():
                                gr.Markdown("插件可读取“输入区”文本/路径作为参数（上传文件自动修正路径）")
                            with gr.Row():
                                for k in crazy_fns:
                                    if not crazy_fns[k].get("AsButton", True): continue
                                    variant = crazy_fns[k]["Color"] if "Color" in crazy_fns[k] else "secondary"
                                    crazy_fns[k]["Button"] = gr.Button(k, variant=variant)
                                    crazy_fns[k]["Button"].style(size="sm")
                         with gr.Row():
                            with gr.Accordion("更多函数插件", open=False):
                                # update
                                dropdown_fn_list = crazy_fns.keys() 
                                with gr.Row():
                                    dropdown = gr.Dropdown(dropdown_fn_list, value=r"打开插件列表", label="", show_label=False).style(container=False)  
                                with gr.Row():      
                                    plugin_advanced_arg = gr.Textbox(show_label=True, label="高级参数输入区", visible=False, 
                                                                 placeholder="特殊函数插件的高级参数输入区").style(container=False)
                                with gr.Row():
                                    switchy_bt = gr.Button(r"请先从插件列表中选择", variant="secondary")
                         with gr.Row():
                            with gr.Accordion("点击展开“文件上传区”。上传本地文件/压缩包供函数插件调用。", open=False) as area_file_up:
                                file_upload = gr.Files(label="任何文件, 但推荐上传压缩文件(zip, tar)", file_count="multiple")        
                       
                    with gr.Column(scale=20):
                         with gr.Accordion('输出区', open=True):
                            with gr.TabItem('图像输出'):   
                                gallery = gr.Gallery(label="Generated images",show_label=False,elem_id="gallery",).style(preview=True, grid=2, object_fit="scale-down")
                            with gr.TabItem('视频输出'):  
                                video_output = gr.Video(label="Generated video", format="mp4").style(width=600)
                         with gr.TabItem('图文理解'):  
                            with gr.Row(): 
                                output_text = gr.Textbox(label="tag2text",lines=2)
                                with gr.Row():
                                        output_tag= gr.outputs.Textbox(label="Tag").style(height=1)  
                            with gr.Row():
                               
                                zh_select=gr.inputs.Checkbox(label='英译中 Tag2Text【选后需重载模型】',default=False).style(width=1) 
                                with gr.Row():
                                    output_classes= gr.Textbox(label="Class Numbers ",lines=1,
                                            placeholder="generate classes numbers,color flag or save_txt must be ture/你必须启动存储txt的功能，这个是全局的").style(conatiner=False,width=1)
                                        
                         with gr.Row():
                            with gr.Accordion("备选输入区", open=True, visible=False) as area_input_secondary:
                                 system_prompt = gr.Textbox(show_label=True, placeholder=f"Chat Prompt", label="下方输入对话支持图像和文本", value="AI assistant.")
                                 #stopBtn2 = gr.Button("停止", variant="secondary"); stopBtn2.style(size="sm")
                                 clearBtn2 = gr.Button("清除", variant="secondary", visible=False); clearBtn2.style(size="sm")           
                         with gr.Row():
                            with gr.Column(scale=2):
                                result_text = gr.Chatbot(label=f'当前模型:{LLM_MODEL}', value=[("", "Hi, What do you want to know ?")]).style(height=CHATBOT_HEIGHT)
                                history = gr.State([])
               
               #Recording_audio.click(fn=toggle_operation,inputs=[asr_select],outputs=[input_text]) # 将 toggle_operation 函数绑定到按钮
                  # 功能区显示开关与功能区的互动
               def fn_area_visibility(a):
                    ret = {}
                    ret.update({area_basic_fn: gr.update(visible=("基础功能区" in a))})
                    ret.update({area_crazy_fn: gr.update(visible=("函数插件区" in a))})
                    ret.update({area_input_primary: gr.update(visible=("底部输入区" not in a))})
                    ret.update({area_input_secondary: gr.update(visible=("底部输入区" in a))})
                    ret.update({clearBtn: gr.update(visible=("输入清除键" in a))})
                    ret.update({clearBtn2: gr.update(visible=("输入清除键" in a))})
                    ret.update({plugin_advanced_arg: gr.update(visible=("插件参数区" in a))})
                    if "底部输入区" in a: ret.update({txt: gr.update(value="")})
                    return ret
               checkboxes.select(fn_area_visibility, [checkboxes], [area_basic_fn, area_crazy_fn, area_input_primary, area_input_secondary, chat_txt,txt , clearBtn, clearBtn2, plugin_advanced_arg] )
               sadtalker_submit.click(fn=sadtalker_demo,inputs=[sadtalker_path,sadtalker_config,image_prompt,upload_audio, preprocess_type,is_still_mode,enhancer,
                                batch_size, size_of_image, pose_style, exp_weight],outputs=[video_output])
               #audio_to_face.click(fn=t2s, inputs=[result_text,input_text,gr.State(True),omniverse_switch], outputs=[upload_audio] )                                 
               asr_button.click(fn=load_speech_model,inputs=[asr_select,tts_select],outputs=[loads_flag])        
               asr.click(fn=s2t, inputs=[upload_audio], outputs=[input_text])                    
               tts.click(fn=t2s, inputs=[input_text,tts_select], outputs=[upload_audio]) 
               # fine tune VisualGLM  
               visualglm_args.append(train_methods)   
               fine_tune.click(fn=train_visualGLM,inputs=visualglm_args,outputs=[txt])   
                 
               # visualGLM inputs    
               cs=[]                 
               cs.extend(list_methods)  
               cs.extend([zh_select, visual_glm,device_input, quant, loads_flag])
               loads_model_button.click(fn=load_auto_backend_models,inputs=cs,outputs=[loads_flag])                     
               inputs.append(zh_select)
               
               def on_md_dropdown_changed(k):
                    return {result_text: gr.update(label="当前模型："+k)}
               md_dropdown.select(on_md_dropdown_changed, [md_dropdown],[result_text])
               
               outputs = [gallery, output_text, output_tag,output_classes]             
               input_combo = [cookies, max_length_sl, md_dropdown,chat_txt,txt,top_p, temperature, result_text, history,system_prompt,plugin_advanced_arg,omniverse_switch,record_audio,asr_gpt,quant_chatglm,chat_app]        
               output_combo = [cookies, result_text, history, status]
              # output_combo2=[result_text, history, status]
               predict_args = dict(fn=ArgsGeneralWrapper(predict_all), inputs=input_combo, outputs=output_combo)  
               chat_args=dict(fn=ArgsGeneralWrapper(talk_all), inputs=input_combo, outputs=output_combo)  
               run_button.click(fn=Auto_run, inputs=inputs, outputs=outputs)
                # 提交按钮、重置按钮
               cancel_handles.append(chat_txt.submit(**predict_args))
               cancel_handles.append(txt.submit(**predict_args))
               cancel_handles.append(run_button_chat.click(**predict_args))
               cancel_handles.append(run_button_2.click(**predict_args))
               cancel_handles.append(chat_app_button.click(**chat_args)) 
               resetBtn.click(lambda: ([], [], "已重置"), None, [result_text, history, status])
               stopBtn2.click(fn=None, inputs=None, outputs=None, cancels=cancel_handles)
               clearBtn.click(lambda: ("",""), None, [chat_txt,txt])
               clearBtn2.click(lambda: ("",""), None, [chat_txt,txt])
               if AUTO_CLEAR_TXT:
                    run_button_chat.click(lambda: ("",""), None, [chat_txt,txt])
                    run_button_2.click(lambda: ("",""), None, [chat_txt,txt])
                    chat_txt.submit(lambda: ("",""), None, [chat_txt,txt])
                    txt.submit(lambda: ("",""), None, [chat_txt,txt])
               for k in functional:
                    if ("Visible" in functional[k]) and (not functional[k]["Visible"]): continue
                    dict_args=dict(fn=ArgsGeneralWrapper(predict_all), inputs=[*input_combo, gr.State(True),gr.State(k)], outputs=output_combo)
                    
                    cancel_handles.append(functional[k]["Button"].click(**dict_args))
               # 文件上传区，接收文件后与chatbot的互动
               file_upload.upload(on_file_uploaded, [file_upload, result_text, chat_txt, txt, checkboxes], [result_text, chat_txt, txt])
                # 函数插件-固定按钮区
               for k in crazy_fns:
                    print(f'检查插件名字{k}，是否载入')
                    if not crazy_fns[k].get("AsButton", True): continue
                    click_handle = crazy_fns[k]["Button"].click(ArgsGeneralWrapper(crazy_fns[k]["Function"]), [*input_combo, gr.State(PORT)], output_combo)
                    click_handle.then(on_report_generated, [cookies, file_upload, result_text], [cookies, file_upload, result_text])
                    cancel_handles.append(click_handle)
                # 函数插件-下拉菜单与随变按钮的互动
               def on_dropdown_changed(k):
                    variant = crazy_fns[k]["Color"] if "Color" in crazy_fns[k] else "secondary"
                    ret = {switchy_bt: gr.update(value=k, variant=variant)}
                    if crazy_fns[k].get("AdvancedArgs", False): # 是否唤起高级插件参数区
                        ret.update({plugin_advanced_arg: gr.update(visible=True,  label=f"插件[{k}]的高级参数说明：" + crazy_fns[k].get("ArgsReminder", [f"没有提供高级参数功能说明"]))})
                    else:
                        ret.update({plugin_advanced_arg: gr.update(visible=False, label=f"插件[{k}]不需要高级参数。")})
                    return ret
               dropdown.select(on_dropdown_changed, [dropdown], [switchy_bt, plugin_advanced_arg] )
               def on_md_dropdown_changed(k):
                    return {result_text: gr.update(label="当前模型："+k)}
               md_dropdown.select(on_md_dropdown_changed, [md_dropdown], [result_text] )
                # 随变按钮的回调函数注册
               def route(request: gr.Request, k, *args, **kwargs):
                    if k in [r"打开插件列表", r"请先从插件列表中选择"]: return
                    yield from ArgsGeneralWrapper(crazy_fns[k]["Function"])(request, *args, **kwargs)
               click_handle = switchy_bt.click(route,[switchy_bt, *input_combo, gr.State(PORT)], output_combo)
               click_handle.then(on_report_generated, [cookies, file_upload, result_text], [cookies, file_upload, result_text])
               cancel_handles.append(click_handle)
                # 终止按钮的回调函数注册
             #  stopBtn.click(fn=None, inputs=None, outputs=None, cancels=cancel_handles)
               stopBtn2.click(fn=None, inputs=None, outputs=None, cancels=cancel_handles)     
               #VisualGLM run         
               run_button_2.click(fn=visual_chat,inputs=[chat_txt, visual_temperature, visual_top_p, image_prompt,
                                                         result_text,record_audio,upload_audio,omniverse_switch],
                                outputs=[txt, result_text])
               prompt_input.submit(fn=visual_chat,inputs=[chat_txt, visual_temperature, visual_top_p, image_prompt,
                                                         result_text,record_audio,upload_audio,omniverse_switch],
                                        outputs=[txt,result_text])
               #upload_audio.upload(fn=clear_fn_image, inputs=clear_button, outputs=[result_text])
               image_prompt.upload(fn=clear_fn_image, inputs=clear_button, outputs=[result_text])
               clear_button.click(lambda: ("","","","",""), None, [prompt_input,result_text,txt, input_text,chat_txt])
               image_prompt.clear(fn=clear_fn_image, inputs=clear_button, outputs=[result_text])
            #    def init_cookie(cookies, chatbot):
            #             # 为每一位访问的用户赋予一个独一无二的uuid编码
            #             cookies.update({'uuid': uuid.uuid4()})
            #             return cookies
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
          auto_opentab_delay(7901)
          block.queue(concurrency_count=CONCURRENT_COUNT).launch(server_name='0.0.0.0', server_port=7901,debug=True, share=False)
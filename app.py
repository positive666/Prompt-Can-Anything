from model_cards.autoback import AutoBackend
import argparse
import os
import platform
import sys
from pathlib import Path
import  numpy as np 
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
from utils.ops import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                     dilate_mask, increment_path, non_max_suppression ,print_args, scale_boxes, xyxy2xywh,save_format)
from utils.plot import Annotator, save_one_box,show_box,show_mask,save_mask_data,Draw_img
from ChatGPT.GPT import Chatbot
from ChatGPT.config.private import API_KEY,PROXIES

from utils.torch_utils import select_device
from utils.conf import SAM_MODEL_TYPE,GROUNED_MODEL_TYPE,Tag2Text_Model_Path,NUM_WORKS
from utils import VID_FORMATS,IMG_FORMATS,write_categories
sys.path.append("VisualGLM_6B")
from VisualGLM_6B.chatglm import *
import multiprocessing
import xml.etree.cElementTree as ET
import gradio as gr
from gradio.inputs import File
import random

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  #  root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
global chatbot
global categories
categories = {}
global category_colors
category_colors={}
# 初始对应类别编号
class_ids = []
models_config = {'tag2text': None, 'lama': None,'sam': None,'grounded': None,'sd': None,'chat_glm': None}
#global memory_model

def auto_opentab_delay(port=7585):
               import threading, webbrowser, time
               print(f"如果浏览器没有自动打开，请复制并转到以下URL：")
               print(f"\t（orgstyle）: http://localhost:{port}, (Darkstyle）: http://localhost:{port}/?__dark-theme=true")
               def open(): 
                    time.sleep(2)                                           # 打开浏览器
                    webbrowser.open_new_tab(f"http://localhost:{port}/?__dark-theme=true")
               threading.Thread(target=open, name="open-browser", daemon=True).start()

def load_auto_backend_models(lama,sam,det,tag2text,device):
    """
    加载多个模型
    """
    # Load model
    
    device = select_device(device)
    if tag2text and not models_config['tag2text']:
        models_config['tag2text'] = AutoBackend("tag2text",weights=Tag2Text_Model_Path,device=device)
    elif not tag2text  :
        models_config['tag2text'] =None 
    else :
        print('tag2text pass')    
    if det and not models_config['grounded']:
        models_config['grounded'] = AutoBackend("grounded-DINO",weights=GROUNED_MODEL_TYPE['S'], device=device,
        args_config= 'model_cards/groundingdino/config/GroundingDINO_SwinT_OGC.py')
    elif not det  :
        models_config['grounded'] =None 
    else :
        print('grounded pass')
        
    if sam and not models_config['sam']:
        models_config['sam']= AutoBackend("segment-anything",weights=SAM_MODEL_TYPE['vit_h'] ,device=device)
    elif not sam :
        models_config['sam'] =None 
    else :
        print(' sam pass')
        
    if lama and not models_config['lama']:
        models_config['lama']= AutoBackend("lama",weights=None,args_config='model_cards/lama/configs/prediction/default.yaml',device=device)
    elif not lama :
        models_config['lama'] =None 
    else :
        print(' lama pass')   

def Auto_run(
        source= 'data/images',  # file/dir/URL/glob, 0 for webcam
        img_input='',
        input_prompt="Anything in this image",
        conf_thres=0.3,  # confidence threshold
        iou_thres=0.5,  # NMS IOU threshold
        text_thres=0.2,
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu   
        save_conf=False,  # save confidences in --save-txt labels
        img_save=False,  # do not save images/videos
        chatgpt=False,
        visualize=False,  # visualize features
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        lama=False,   # use lama models
        sam=True,    # use segment-anythings
        det=True,    # use grounded detect model with text
        tag2text=True,
        save_txt=False,  # save results to *.txt
        save_xml=False,  # save results to *.xml
        save_mask=False,
        save_caption=False,
        batch_process=False,
        color_flag=False,
        process_name=0,
        ):  

            global models_config
            global category_colors
            # if not memory_model  
            load_auto_backend_models(lama,sam,det,tag2text,device)
            # memory_model=True
            LOGGER.info (f'proceess ID：{process_name},loads model list ：{models_config.keys()}')
            if chatgpt:
               # global chatbot
                chatbot = Chatbot(api_key=API_KEY,proxy=PROXIES,engine="gpt-3.5-turbo")
            cls_index = -1        # 设置默认值为 -1
            if img_input:
                source =img_input
            source = str(source)
            
            print(f'input:{source}')
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
                        print(f'text_prompt:{prompt}')
                        preds = models_config['tag2text'](im = img_rgb ,prompt=prompt,box_threshold=conf_thres,text_threshold=text_thres,iou_threshold=iou_thres)
                    # Currently ", " is better for detecting single tags
                    # while ". " is a little worse in some case
                        prompt=preds[0].replace(' |', ',')
                        caption=preds[2]
                        print(f"Caption: {caption}")
                        print(f"Tags: {prompt}")
                        if save_caption:
                            save_format(label_format="txt",save_path=f'{save_dir}/captions',img_name=name_p, results=caption)
                    if det:
                        if input_prompt:
                            prompt=input_prompt
                            print('your input prompt replace default:',prompt)
                        preds= models_config['grounded'](im = img_rgb,prompt=prompt, box_threshold=conf_thres,text_threshold=text_thres, iou_threshold=iou_thres) 
                        if chatgpt:
                            from gpt_demo import check_caption
                            caption=check_caption(caption, preds[2], chatbot)
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
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}/captions")
            if save_mask:
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}/masks")
            LOGGER.info('Done...')
            #re=Image.open(f'{save_dir}/{seen}.jpg') 

            return [[img_rgb],caption,prompt,len(categories)]
            


def main(args):
      
          check_requirements(exclude=('tensorboard', 'thop'))
          global models_config,tokenizer_glm
          if args.chat_glm:
                 models_config['chat_glm']=VisualGLM(quant=args.quant)
          block=gr.Blocks()
          inputxs=[]
          outputs=[]
          block=block.queue()
          with block:
               with gr.Row():
                    with gr.Column():
                         with gr.Accordion('Grounded-DINO threshold Options', open=False):
                            box_threshold=gr.inputs.Number(label='Confidence Threshold', default=0.3)
                            iou_threshold=gr.inputs.Number(label='ioue Threshold', default=0.5)
                            text_threshold=gr.inputs.Number(label='TEXT Threshold', default=0.25)
                            device_input=gr.inputs.Textbox(label='device',default='0')                             
                         with gr.Accordion('others Options', open=False):
                            option_inputs  = {
                            'Save Conf': gr.inputs.Checkbox(label='Save Conf',default=False),
                            'Save img': gr.inputs.Checkbox(label='Save img',default=False),
                            'Chat GPT': gr.inputs.Checkbox(label='ChatGPT',default=False),
                            'Visualize': gr.inputs.Checkbox(label='Visualize',default=False),
                            'Project': gr.inputs.Textbox(label='Project:save dir_path',default='runs/detect'),
                            'Name': gr.inputs.Textbox(label='Name',default='exp'),
                            'Exist Ok': gr.inputs.Checkbox(label='Exist Ok',default=False)
                            }   
                           
                         inputxs.extend(list(option_inputs.values()))
                         with gr.Accordion('Method_Options:free combo', open=True):                
                                   
                            methods_options={'Lama': gr.inputs.Checkbox(label='Lama model',default=False), 'Sam': gr.inputs.Checkbox(label='Sam model',default=False),
                                'Det': gr.inputs.Checkbox(label='Grounded',default=False), 
                                'Tag2text': gr.inputs.Checkbox(label='Tag2text',default=False), 
                            }
                    
                         inputxs.extend(list( methods_options.values()))      
                         with gr.Accordion('format Options', open=False):                
                                   
                                save_options={
                                'Save txt': gr.inputs.Checkbox(label='Save txt',default=False), 
                                'Save xml': gr.inputs.Checkbox(label='Save xml',default=False), 
                                'Save Mask': gr.inputs.Checkbox(label='Save Mask',default=False),  
                                'Save Caption': gr.inputs.Checkbox(label='Save Caption',default=False),  
                                'Batch Process': gr.inputs.Checkbox(label='Batch Process',default=False), 
                                'Color Flag': gr.inputs.Checkbox(label='Color Flag : classes mask',default=False)
                            }
                         inputxs.extend(list( save_options.values()))
                         
                         API_KEY=gr.inputs.Textbox(label='OPENAI_kety',default='')  
                         dir_inputs =gr.inputs.Textbox(label='dir_path',default='train_imgs')
                         prompt_input=gr.inputs.Textbox(lines=3, label="Prompt: User Specified Tags (Optional, Enter with commas)")
                         run_button = gr.Button('Run')
                         image_prompt = gr.Image(type="filepath", label="Image Prompt", value=None)
                         inputs = [dir_inputs,image_prompt,prompt_input,box_threshold,iou_threshold,text_threshold,device_input]
                         inputs.extend(inputxs)
                         
                         if models_config['chat_glm']:
                            with gr.Row():
                                run_button_2 = gr.Button('Send')
                                clear_button = gr.Button('Clear')
                            with gr.Row():
                                temperature = gr.Slider(maximum=1, value=0.8, minimum=0, label='Temperature')
                                top_p = gr.Slider(maximum=1, value=0.4, minimum=0, label='Top P')
                            with gr.Group():
                                with gr.Row():
                                    maintenance_notice = gr.Markdown(MAINTENANCE_NOTICE1)   
                    with gr.Column(scale=1.5):
                         gallery = gr.Gallery(label="Generated images",show_label=False,elem_id="gallery",).style(preview=True, grid=2, object_fit="scale-down")
                         output_text = gr.Textbox(label="Caption",lines=2)
                         output_classes= gr.Textbox(label="Class_numbers:auto generate classes numbers, 【color flag】 or 【save_txt】 must be ture ")
                         output_tag= gr.outputs.Textbox(label="Tag")
                         outputs = [gallery, output_text, output_tag,output_classes]
                         if models_config['chat_glm']:   
                            result_text = gr.components.Chatbot(label='Multi-round conversation History', value=[("", "Hi, What do you want to know about this image?")]).style(height=550)
               
               run_button.click(fn=Auto_run, inputs=inputs, outputs=outputs)
               if  models_config['chat_glm']:   
                            
                run_button_2.click(fn=models_config['chat_glm'].request_model,inputs=[prompt_input, temperature, top_p, image_prompt, result_text],
                            outputs=[prompt_input, result_text])
                prompt_input.submit(fn=models_config['chat_glm'].request_model,inputs=[prompt_input, temperature, top_p, image_prompt, result_text],
                                    outputs=[prompt_input, result_text])
                clear_button.click(fn=clear_fn, inputs=clear_button, outputs=[prompt_input, result_text, image_prompt])
                image_prompt.upload(fn=clear_fn2, inputs=clear_button, outputs=[result_text])
                image_prompt.clear(fn=clear_fn2, inputs=clear_button, outputs=[result_text])
        
          auto_opentab_delay()
          block.queue(concurrency_count=100)
          block.launch(server_name='0.0.0.0', server_port=7585, debug=True, share=False)
     
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", choices=[8, 4], type=int, default=4)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--chat_glm", default=False,action="store_true")
    args = parser.parse_args()
    main(args)
    
     
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
from PIL import Image
import random
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  #  root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.ops import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                     dilate_mask, increment_path, non_max_suppression ,print_args, scale_boxes, xyxy2xywh,save_format)
from utils.plot import Annotator, save_one_box,show_box,show_mask,save_mask_data
from ChatGPT.GPT import Chatbot
from ChatGPT.config.private import API_KEY,PROXIES

from utils.torch_utils import select_device
from utils.conf import SAM_MODEL_TYPE,GROUNED_MODEL_TYPE,Tag2Text_Model_Path,NUM_WORKS
from utils import VID_FORMATS,IMG_FORMATS,write_categories

import multiprocessing
import xml.etree.cElementTree as ET
from tqdm import tqdm

# 初始已知类别列表
global categories
categories = {}
global category_colors
category_colors={}
# 初始对应类别编号
class_ids = []
models_config = {'tag2text': None, 'lama': None,'sam': None,'grounded': None,'sd': None}

def load_auto_backend_models(opt):
    """
    加载多个模型
    """
    # Load model
    device = select_device(opt.device)
    if opt.tag2text:
        models_config['tag2text'] = AutoBackend("tag2text",weights=Tag2Text_Model_Path,device=device, fp16=opt.half)
    if opt.det:
        models_config['grounded'] = AutoBackend("grounded-DINO",weights=GROUNED_MODEL_TYPE['S'], device=device,
        args_config= 'model_cards/groundingdino/config/GroundingDINO_SwinT_OGC.py', fp16=opt.half)
    if opt.sam:
        models_config['sam']= AutoBackend("segment-anything",weights=SAM_MODEL_TYPE['vit_h'] ,device=device, fp16=opt.half)
    if opt.lama:
        models_config['lama']= AutoBackend("lama",weights=None,args_config='model_cards/lama/configs/prediction/default.yaml',device=device)
    #return models_config
        
def Auto_run(weights=ROOT / '',  # model.pt path(s)
        source= 'data/images',  # file/dir/URL/glob, 0 for webcam
        input_prompt="Anything in this image",
        data=ROOT / 'data/',  # dataset.yaml path
        imgsz=(1920, 1080),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        text_thres=0.3,
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_xml=False,  # save results to *.xml
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        chatgpt=False,
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        trace=False,  # u
        lama=False,   # use lama models
        sam=True,    # use segment-anythings
        det=True,    # use grounded detect model with text
        tag2text=True,
        save_mask=False,
        save_caption=False,
        batch_process=False,
        color_flag=False,
        process_name=0,
        ):  
            LOGGER.info(f'当前的进程ID：{process_name},加载的模型列表：{models_config.keys()}')
            cls_index = -1      # 设置默认值为 -1
            source = str(source)
            print(f'input:{source}')
            img_paths=None
            if os.path.isdir(source):
                img_paths = [os.path.join(source, f) for f in os.listdir(source) if
                    Path(f).suffix[1:] in (IMG_FORMATS + VID_FORMATS)] 
            # 获取文件夹中的所有图像
            is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
            save_img = not nosave and not source.endswith('.txt')  # save inference images
            is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
            webcam = source.isnumeric() or source.endswith('.streams') or (is_url )
            if is_url and is_file:
                source = check_file(source)  # download
            # Directories
            save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
            (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
            (save_dir / 'xmls' if save_xml else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
            (save_dir / 'masks' if save_mask else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
            (save_dir / 'captions' if save_caption else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
            p = Path(str(save_dir) )  # to Path
            #save_path = str(save_dir / p.name)  # im.jpg    
           # txt_path  = str(save_dir / 'labels' / p.stem) + ''  # im.txt
            seen=0
            # loda data and inference
            caption=None
            for source in tqdm(img_paths,desc="Processing"): 
                    im = cv2.imread(source)
                    name_p= source.split('/')[-1].split('.')[0]
                    img_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    preds=None
                    masks=[]
                    prompt=input_prompt
                    if tag2text:
                        preds = models_config['tag2text'](im=img_rgb ,prompt=prompt)
                    # Currently ", " is better for detecting single tags
                    # while ". " is a little worse in some case
                        prompt=preds[0].replace(' |', ',')
                        caption=preds[2]
                        print(f"Caption: {caption}")
                        print(f"Tags: {prompt}")
                        if save_caption:
                            save_format(label_format="txt",save_path=f'{save_dir}/captions', 
                            img_name=name_p, results=caption)
                    if det:
                        preds= models_config['grounded'](im= img_rgb,prompt= prompt) 
                        if chatgpt:
                            from gpt_demo import check_caption
                            caption=check_caption(caption,preds[2])
                    if preds is not None and sam:
                            masks= models_config['sam'](im= img_rgb, prompt=preds[0])
                            if  save_mask:
                                #save_coco_segmentation_txt(masks,name_p,f'{save_dir}/masks',preds[0],preds[2])
                                save_mask_data(str(save_dir)+'/masks', caption, masks, preds[0], preds[2],name_p)
                    # Write results
                    if save_img:
                        seen+=1
                        plt.figure(figsize=(10,10))
                        plt.imshow(img_rgb)
                        for box,sc,label in zip(preds[0],preds[1],preds[2]):
                                show_box(box.numpy(),plt.gca(),label)
                        for mask in masks:         
                                show_mask(mask.cpu().numpy(),plt.gca(),random_color=True)
                        plt.title('Captioning: ' + caption + '\n' + 'Tagging:' + prompt + '\n')    
                        plt.axis('off')
                        plt.savefig(f'{save_dir}/{seen}.png',bbox_iches='tight',dpi=300,pad_inches=0.0)     
                        
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
                    
                    if color_flag or save_txt:
                        seg_mask = np.zeros_like(img_rgb)  # img_array 为输入图像的数组表示
                        category_color=[]
                        for xyxy, conf, cls,mask in zip(preds[0],preds[1],preds[2],masks):       #per im boxes              
                                xywh = (xyxy2xywh((xyxy).view(1,4)) / gn).view(-1).tolist()  # normalized xywh   
                                if cls not in categories:
                                # print(f'Add {cls} to categories: {categories}')
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
                    if color_flag and save_mask:
                            plt.figure(figsize=(10,10))
                            plt.imshow(seg_mask)
                            plt.title('Captioning: ' + caption + '\n' + 'Tagging:' + prompt + '\n')    
                            plt.axis('off')            
                            plt.savefig(os.path.join(f'{save_dir}/masks', f'{name_p}_cls.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)                                                         
                    if save_xml:    
                            h,w=im.shape[:2]
                            save_format("xml",f'{save_dir}/xmls' ,name_p, Path(source).parent,
                                preds, h,w)
            if save_txt:
                #class_ids.append(cls) 
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}/labels")  
            if save_xml:           
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}/xmls")
            if save_caption:
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}/captions")
            if save_mask:
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}/masks")
                   
def run_do(shared_args,process_name=0):
    
    Auto_run(**vars(shared_args), process_name=process_name)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'your model path', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--input_prompt', type=str, default='', help='provide prompt words')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--text-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-xml', action='store_true', help='save results to *.xml')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--chatgpt', action='store_true', help='gpt3.5/4')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--trace', action='store_true', help='trace model')
    parser.add_argument('--lama',default=False, action='store_true', help='lama model')
    parser.add_argument('--sam', default=True,action='store_true', help='seg model')
    parser.add_argument('--det',default=True, action='store_true', help='det model')
    parser.add_argument('--tag2text', default=True,action='store_true', help='tag2text model ')
    parser.add_argument('--save-mask', default=True,action='store_true', help='mask save json')
    parser.add_argument('--save-caption', default=True,action='store_true', help='caption ')
    parser.add_argument('--batch-process', action='store_true', help='therads process file')
    parser.add_argument('--color-flag', action='store_true', help='class-color ')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt

import threading
import concurrent.futures
def main(opt):
    
    check_requirements(exclude=('tensorboard', 'thop'))
    global models_config  
    if opt.chatgpt:
        global chatbot
        chatbot=Chatbot(api_key=API_KEY,proxy=PROXIES,engine="gpt-3.5-turbo")
    # if  not opt.tag2text::
        # LOGGER.info('your must input prompt')
        # words_name= input("please your prompt words: ")
        # opt.input_prompt=words_name
        
    load_auto_backend_models(opt)
    LOGGER.info(f"模型加载成功{models_config.keys()}")
    if opt.batch_process and os.path.isdir(opt.source):
        #检查目录是否存在以及检查是否为目录的操作
        if not os.path.exists(opt.source):
            LOGGER.info(f"Error: Input directory {opt.source} does not exist.")
            return 
        seen=0
        output_dir=f'{opt.source}_subs{seen}'
        segment_size =100
        for file_name in opt.source:
            file_path = os.path.join(opt.source, file_name)
            # pass 
            if not  Path(file_path).suffix[1:] in (IMG_FORMATS + VID_FORMATS):
                continue
            # 使用Pillow库读取图像文件并将其转换为NumPy数组
            img = Image.open(file_path)
            img_array = np.asarray(img)

            # 多线程处理每个图像段
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []  # 用于保存每个线程的未来对象

                # 分段并发读取并进行处理
                for i in range(0, img_array.shape[0], segment_size):
                    start_row = i
                    end_row = min(i + segment_size, img_array.shape[0])
                    future = executor.submit(run_do, img_array, start_row, end_row)
                    futures.append(future)

                # 获取所有未来对象的结果
                for future in concurrent.futures.as_completed(futures):
                    segment = future.result()
    else:
        Auto_run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
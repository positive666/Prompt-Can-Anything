import torch
import json
import math
import platform
import warnings
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import yaml
import torch.nn.functional as F
from PIL import Image

import sys
#Grounding
from model_cards.groundingdino.models import build_model
import  model_cards.groundingdino.datasets.transforms as T
from model_cards.groundingdino.models import build_model
from model_cards.groundingdino.util import box_ops
from model_cards.groundingdino.util.slconfig import SLConfig
from model_cards.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

sys.path.append("model_cards")
sys.path.append('model_cards/Tag2Text')
sys.path.append('model_cards/lama')

from model_cards.Tag2Text import inference 
from Tag2Text.models import tag2text
import torchvision.transforms as TS
from utils.conf import LAMA_MODEL_PATH
# segment anything
from model_cards.segment_anything import build_sam, SamPredictor 

# lama 
from model_cards.lama.saicinpainting.evaluation.utils import move_to_device
from model_cards.lama.saicinpainting.training.trainers import load_checkpoint
from model_cards.lama.saicinpainting.evaluation.data import pad_tensor_to_modulo


 
import numpy as np
import matplotlib.pyplot as plt
from utils import check_requirements,check_suffix,IMAGENET_MEAN,IMAGENET_STD
from utils.downloads import is_url,LOGGER,attempt_download
from torchvision.ops import nms as NMS
from utils.plot import *
Model_CARDS=['lama','segment-anything','grounded-DINO','Tag2Text']


def preprocess_image(img):

    # Convert image from BGR to RGB format using OpenCV library
    #img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = Image.fromarray(img)
    # Define image transformation pipeline using PyTorch library
    transform = T.Compose([
        
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    # Apply image transformation pipeline to numpy array to get PyTorch tensor
    img_tensor,_ = transform(img_array,None)

    return  img_tensor
               

class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = torch.cat(y, 2)  # nms ensemble
        return y, None  # inference, train output

def torch_safe_load(weight):
    """
    This function attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it
    catches the error, logs a warning message, and attempts to install the missing module via the check_requirements()
    function. After installation, the function again attempts to load the model using torch.load().
    Args:
        weight (str): The file path of the PyTorch model.
    Returns:
        The loaded PyTorch model.
    """
    
    check_suffix(file=weight,suffix='.pt')
    file = attempt_download(weight)  # search online if missing locally
    try:
        return torch.load(file, map_location='cpu'),file  # load
    except ModuleNotFoundError as e:
        if e.name == 'models':  # e.name is missing module name
            LOGGER.warning(f"WARNING ⚠️ {weight} requires {e.name}, which is not in ultralytics requirements."
                           f"\nAutoInstall will run now for {e.name} but this feature will be removed in the future."
                           f"\nRecommend fixes are to train a new model using updated ultraltyics package or to "
                        )
        check_requirements(e.name)  # install missing module ,select no
        return torch.load(file, map_location='cpu'), file  # load


def is_similar_string(string):
    for s in Model_CARDS:
        if string.lower() in s.lower():
            print('get your method')
            return string.lower()
    return None

def attempt_load(weights, device=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
     model=[]
    # model=Ensemble()
     for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w),map_location='cpu')# load ckpt
         
        #args = {**DEFAULT_CFG_DICT, **ckpt['train_args']}  # combine model and default args, preferring model args
        # ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model
        # ckpt.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
        ckpt.pt_path = weights  # attach *.pt file path to model
        model.append(ckpt.eval())  # fused or un-fused model in eval mode

     return model  # return ensemble

class AutoBackend(nn.Module):
    # for python inference on various Models
    def __init__(self, methods: str ,weights: None , device=torch.device('cpu'), args_config: str= 'model_cards/groundingdino/config/GroundingDINO_SwinT_OGC.py', fp16: bool=False,num_classes:int=1,tag2text_thres=0.6):
         # Usage:
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   Model methos Card 
        
        from utils.downloads import attempt_download
        super().__init__()
        self.flag=is_similar_string(methods)
       
        self.device=device
        self.nc=num_classes
        
        w = str(weights[0] if isinstance(weights, list) else weights)
        nn_module=isinstance(weights,torch.nn.Module)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
       # w = attempt_download(w)  # download if not local
        fp16 &= pt or jit or onnx or engine  or nn_module # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
        self.model,metadata=None,None 
        cuda = torch.cuda.is_available() and self.device.type != 'cpu'  # use CUDA
        
        if pt or weights is None :  # PyTorch
            if methods == "grounded-DINO":
               config_args=SLConfig.fromfile(args_config)
               config_args.device=self.device
               model=build_model(config_args)
               ckpt=torch.load(w)
               load_res = model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
               self.flag=methods
               self.model = model.to(self.device).eval()  # explicitly assign for to(), cpu(), cuda(), half()
              
              
            elif methods== "segment-anything":
                print("-----init Sam------")
                self.model = SamPredictor(build_sam(checkpoint=w).to(self.device))
                
            elif methods == 'lama':
                from omegaconf import OmegaConf
                print('---init lama------') 
                self.predict_config = OmegaConf.load(args_config)
                self.predict_config.model.path = LAMA_MODEL_PATH
        
                train_config_path = os.path.join(
                self.predict_config .model.path, 'config.yaml')

                with open(train_config_path, 'r') as f:
                    train_config = OmegaConf.create(yaml.safe_load(f))

                train_config.training_model.predict_only = True
                train_config.visualizer.kind = 'noop'

                checkpoint_path = os.path.join(
                    self.predict_config .model.path, 'models',
                    self.predict_config .model.checkpoint
                )
                self.model = load_checkpoint(
                    train_config, checkpoint_path, strict=False, map_location='cpu')
                self.model.freeze()
                if not self.predict_config .get('refine', False):
                    self.model.to(self.device)
                
            elif methods == 'tag2text':
                print('----Init Tag2Text----')
                 # initialize Tag2Text
                normalize = TS.Normalize(mean=IMAGENET_MEAN,
                                        std=IMAGENET_STD,
                                        )
                self.size=384
                self.transform = TS.Compose([
                                TS.Resize((self.size, self.size)),
                                TS.ToTensor(), normalize
                            ])
                # filter out attributes and action categories which are difficult to grounding
                delete_tag_index = []
                for i in range(3012, 3429):
                    delete_tag_index.append(i) 
                # load model
                self.model = tag2text.tag2text_caption(pretrained=w,
                                image_size=self.size,
                                vit='swin_b',
                                delete_tag_index=delete_tag_index).to(self.device).eval()    
            else :
                LOGGER('not find methods')
                raise TypeError(f'model=:{methods} is not a  saved method')
                
            
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            cuda = torch.cuda.is_available()
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if 'stride' in meta:
                stride, names = int(meta['stride']), eval(meta['names'])
       
        else:
            raise TypeError(f'model=:{w} is not a supported format')

        self.flag=methods
        self.__dict__.update(locals())  # assign all variables to self

    @torch.no_grad()
    def forward(self, im, augment=False, visualize=False,prompt= None ,box_threshold=0.3,text_threshold=0.25, iou_threshold=0.5,boexes_filt=None,with_logits=True):
        #  inference
            H,W=im.shape[0],im.shape[1]
            if self.fp16 and im.dtype != torch.float16:
                im = im.half()  # to FP16     
            if self.flag=="grounded-DINO":
                    
                    caption=prompt
                    input_tensor=preprocess_image(im).to("cuda:0")
                    caption = caption.lower()
                    caption = caption.strip()
                    if not caption.endswith("."):
                            caption = caption + "."
                    with torch.no_grad():
                            y = self.model(input_tensor[None], captions=[caption])
                    logits = y["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
                    boxes = y["pred_boxes"].cpu()[0]  # (nq, 4)
        
                    # filter output
                    logits_filt = logits.clone()
                    boxes_filt = boxes.clone()
                    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
                    logits_filt = logits_filt[filt_mask]  # num_filt, 256
                    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
                    
                    # get phrase
                    tokenlizer = self.model.tokenizer
                    tokenized = tokenlizer(caption)
                    # build pred
                    pred_phrases = []
                    scores=[]
        
                    for logit, box in zip(logits_filt, boxes_filt):
                            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
                            pred_phrases.append(pred_phrase)  
                            scores.append(logit.max().item())
                            box[:]= box * torch.Tensor([W,H,W,H])
                            box[:2]-= box[2:]/2
                            box[2:]+= box[:2]
                    #NMS
                    id_nms= NMS(boxes_filt,torch.tensor(scores),iou_threshold).cpu().numpy().tolist()
                    boxes_filt=boxes_filt[id_nms]
                    scores=[scores[i] for i in id_nms]
                    pred_phrases=[pred_phrases[i] for i in id_nms]
                    return [boxes_filt, torch.Tensor(scores), pred_phrases]
                
            elif self.flag == "segment-anything":
                    
                   # im=cv2.cvtColor(im,cv2.COLOR_BAYER_BG2RGB)
                    self.model.set_image(im)
                    boxes = self.model.transform.apply_boxes_torch(prompt, (H,W)).to(self.device) 
                    masks, _, _ = self.model.predict_torch(
                    point_coords = None,
                    point_labels = None,
                    boxes = boxes.to(self.device),
                    multimask_output = False,
                )
                    return masks
            elif self.flag=="tag2text":
                    print('tag2text inference output')
                    raw_image = cv2.resize(im, (self.size, self.size))
                    raw_image=Image.fromarray(raw_image)
                    raw_image  = self.transform(raw_image).unsqueeze(0).to(self.device)
                    return inference.inference(raw_image , self.model, prompt)

            elif self.flag=='lama':
                
                mask=prompt
                assert len(mask.shape) == 2
                if np.max(mask) == 1:
                        mask = mask * 255
                img = torch.from_numpy(im).float().div(255.)
                mask = torch.from_numpy(mask).float()
                batch = {}
                batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
                batch['mask'] = mask[None, None]
                unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
                batch['image'] = pad_tensor_to_modulo(batch['image'], 8)
                batch['mask'] = pad_tensor_to_modulo(batch['mask'], 8)
                batch = move_to_device(batch, self.device)
                batch['mask'] = (batch['mask'] > 0) * 1

                batch = self.model(batch)
                cur_res = batch[self.predict_config.out_key][0].permute(1, 2, 0)
                cur_res = cur_res.detach().cpu().numpy()

                if unpad_to_size is not None:
                    orig_height, orig_width = unpad_to_size
                    cur_res = cur_res[:orig_height, :orig_width]

                cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
                return cur_res
                            
            elif self.onnx:  # ONNX Runtime
                im = im.cpu().numpy()  # torch to numpy
                y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
                if isinstance(y, (list, tuple)):
                    return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
                else:
                    return self.from_numpy(y)
          
   
    @staticmethod
    def _model_type(p='path/to/model.pt'):
    # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
    # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]

        def export_formats():
            x = [
                ['PyTorch', '-', '.pth' ,True, True],
                ['TorchScript', 'torchscript', '.torchscript', True, True],
                ['ONNX', 'onnx', '.onnx', True, True],
                ['OpenVINO', 'openvino', '_openvino_model', True, False],
                ['TensorRT', 'engine', '.engine', False, True],
                ['CoreML', 'coreml', '.mlmodel', True, False],
                ['CKPT', '.ckpt', 'ckpt', True, True],
                ['TensorFlow GraphDef', 'pb', '.pb', True, True],
                ['TensorFlow Lite', 'tflite', '.tflite', True, False],
                ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', False, False],
                ['TensorFlow.js', 'tfjs', '_web_model', False, False],
                ['PaddlePaddle', 'paddle', '_paddle_model', True, True],
            ]
            return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])

        sf = list(export_formats().Suffix)  # export suffixes

        if not is_url(p, False):
            check_suffix(Path(p).name, sf)  # check for suffix

        url = urlparse(p)  # if url, check if Triton inference server
        types = [suffix in Path(p).name for suffix in sf]
        types[8] &= not types[9]  # If tflite, make sure not edgedpu
        triton = not any(types) and all(scheme in url.scheme for scheme in ['http', 'grpc']) and url.netloc  # check if Triton inference server
        return types + [triton]
   
   
   


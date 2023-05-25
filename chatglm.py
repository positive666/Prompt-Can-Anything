import gradio as gr
from PIL import Image
import os
import json
from VisualGLM.model import is_chinese, get_infer_setting, generate_input, chat
import torch
#!/usr/bin/env python

from VisualGLM.model import is_chinese, get_infer_setting, generate_input, chat
import torch
#!/usr/bin/env python
class VisualGLM:
    def __init__(self,gpu_device=0,quant=4):
        self.model,self.tokenizer = get_infer_setting(gpu_device, quant)
        

    def generate_text_with_image(self, input_text, image, history=[], request_data=dict(), is_zh=True):
        input_para = {
            "max_length": 2048,
            "min_length": 50,
            "temperature": 0.8,
            "top_p": 0.4,
            "top_k": 100,
            "repetition_penalty": 1.2
        }
        input_para.update(request_data)

        input_data = generate_input(input_text, image, history, input_para, image_is_encoded=False)
        input_image, gen_kwargs =  input_data['input_image'], input_data['gen_kwargs']
        with torch.no_grad():
            answer, history, _ = chat(None, self.model, self.tokenizer, input_text, history=history, image=input_image, \
                                max_length=gen_kwargs['max_length'], top_p=gen_kwargs['top_p'], \
                                top_k = gen_kwargs['top_k'], temperature=gen_kwargs['temperature'], english=not is_zh)
        return answer

    def request_model(self, input_text, temperature, top_p, image_prompt, result_previous):
        result_text = [(ele[0], ele[1]) for ele in result_previous]
        for i in range(len(result_text)-1, -1, -1):
            if result_text[i][0] == "" or result_text[i][1] == "":
                del result_text[i]
        print(f"history {result_text}")

        is_zh = is_chinese(input_text)
        if image_prompt is None:
            if is_zh:
                result_text.append((input_text, '图片为空！请上传图片并重试。'))
            else:
                result_text.append((input_text, 'Image empty! Please upload a image and retry.'))
            return input_text, result_text
        elif input_text == "":
            result_text.append((input_text, 'Text empty! Please enter text and retry.'))
            return "", result_text                

        request_para = {"temperature": temperature, "top_p": top_p}
        image = Image.open(image_prompt)
        try:
            answer = self.generate_text_with_image(input_text, image, result_text.copy(), request_para, is_zh)
        except Exception as e:
            print(f"error: {e}")
            if is_zh:
                result_text.append((input_text, '超时！请稍等几分钟再重试。'))
            else:
                result_text.append((input_text, 'Timeout! Please wait a few minutes and retry.'))
            return "", result_text

        result_text.append((input_text, answer))
        print(result_text)
        return "", result_text



DESCRIPTION = '''# <a href="https://github.com/THUDM/VisualGLM-6B">VisualGLM</a>'''

MAINTENANCE_NOTICE1 = 'Hint 1: If the app report "Something went wrong, connection error out", please turn off your proxy and retry.\nHint 2: If you upload a large size of image like 10MB, it may take some time to upload and process. Please be patient and wait.'
MAINTENANCE_NOTICE2 = '提示1: 如果应用报了“Something went wrong, connection error out”的错误，请关闭代理并重试。\n提示2: 如果你上传了很大的图片，比如10MB大小，那将需要一些时间来上传和处理，请耐心等待。'

NOTES = 'This app is adapted from <a href="https://github.com/THUDM/VisualGLM-6B">https://github.com/THUDM/VisualGLM-6B</a>. It would be recommended to check out the repo if you want to see the detail of our model and training process.'


def clear_fn(value):
    return "", [("", "Hi, What do you want to know about this image?")], None

def clear_fn2(value):
    return [("", "Hi, What do you want to know about this image?")]


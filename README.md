# Prompt-Can-Anything
A fully automated  toolkit: You just give prompt ！you only click once! you can do anything by sota model with prompt and creativity

**Motivation**

Current: Making a fully automated AI tool for engineering and research to create Data engines may require the use of more CLIP models

Target:  Plan to generate high-quality data annotation data and train our  models.

So it's just a tool for prompt any thing(YOCO)

 1. Auto-label tool  ,current structure (YOCO)

    In addition, we will introduce video, audio, and 3D annotation in the future.

![structure](asset/data_engine.png)




 2.  Semi-automatic interaction  UI tool (coming soon)

## Feature

- 🔥Data Engine
	
	Provide fully automated data annotation with one-click export (detection, segmentation, text, and nerf  reconstruction results) and refine these through engineering optimization, ,through the correlation models of stable diffusion and gpt, we can create more data source power for downstream tasks. 
	
	</details >
	
- Extended one-click annotation training for the use of three-party projects, such as Yolo, Lora modes. （coming soon）

- Accelerated  processing of videos and datasets（coming soon）
	

<details open >
<summary>⭐ Research🚀 project🔥 Inspiration（In preparation）</summary>

	  At research level, Zero-shot comparative learning is research trend, we hope to understand as much as possible the model design details of the project we are applying, so that we want to combine text, images, and audio to design a strong aligned backbone.
	  At project level, Tensorrt acceleration of the basic model accelerates efficiency.

</details >



### <div align="left">⭐[news list] </div>
	-【2023/5/7】   add web demo:At present, the function of text generation, detection and segmentation of images or image folders on the website has been tested normally, and the program does not need to be restarted, and the last model loading configuration is remembered, and it will be continuously optimized in the future.
	-【2023/5/4】   add  semantic segmentatio label, add args(--color-flag --save-mask )
	-【2023/4/26】  YOCO,Automatic annotation TOOLS:Commit preliminary code ,For the input image or folder, you can obtain the results of detection, segmentation, and text annotation , optional chatgpt api.



**Preliminary-Works**



- [Segment Anything](https://github.com/facebookresearch/segment-anything) : Strong segmentation model. But it needs prompts (like boxes/points) to generate masks. 

- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) :  Strong zero-shot detector which is capable of to generate high quality boxes and labels with free-form text. 

- [Stable-Diffusion](https://github.com/CompVis/stable-diffusion) :  Amazing strong text-to-image diffusion model.

- [Tag2text](https://github.com/xinyu1205/Tag2Text) : Efficient and controllable vision-language model which can simultaneously output superior image captioning and image tagging.
  
- [lama](https://github.com/advimman/lama) :  Resolution-robust large mask Inpainting with Fourier Convolutions

  

**:hammer_and_wrench: YOCO:Quick Start**

First, Make sure you have a basic gpu deep learning environment.

 (Linux is recommended, Windows may have problems compiling Grounded-DINO Deformable- transformer operator, see [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) )

```bash
gir clone https://github.com/positive666/Prompt-Can-Anything
cd Prompt-Can-Anything
```

Install environment:

```bash
pip install -e .
```


Install diffusers（Optional）

```bash
pip install --upgrade diffusers[torch]
```

more ,you can see "pip install < your missing packages>"

Run	

1. downloads models weights

   <!-- insert a table -->

	<table>
	  <thead>
	    <tr style="text-align: left;">
	      <th></th>
	      <th>name</th>
	       <th>backbone</th>
	      <th>Data</th>
	      <th>Checkpoint</th>
	        <th>model-config</th>
	    </tr>
	  </thead>
	  <tbody>
	    <tr>
	      <th>1</th>
	      <td>Tag2Text-Swin</td>
	      <td>Swin-Base</td>
	      <td>COCO, VG, SBU, CC-3M, CC-12M</td>
	      <td><a href="https://huggingface.co/spaces/xinyu1205/Tag2Text/blob/main/tag2text_swin_14m.pth">Download  link</a></td>
	    <tr>
	      <th>2</th>
	      <td>Segment-anything</td>
	       <td>vit</td>
	        <td> </td>
	        <td><a href="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth">Download  link</a>| <a 
	<td><a href="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth">Download  link</a>| <a 
	    <td><a href="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth">Download  link</a></td>
	    <tr>
	      <th>3</th>
	      <td>Lama</td>
	        <td>FFC</td>
	         <td> </td>
	      <td><a href="https://disk.yandex.ru/d/ouP6l8VJ0HpMZg">Download  link</a></td>
	    <tr>
	      <th>4</th>
	      <td>GroundingDINO-T</td>
	      <td>Swin-T</td>
	      <td>O365,GoldG,Cap4M</td>
	      <td><a href="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth">Github link</a> | <a href="https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth">HF link</a></td>
	      <td><a href="https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/GroundingDINO_SwinT_OGC.py">link</a></td>
	    </tr>
	    <tr>
	      <th>5</th>
	      <td>GroundingDINO-B</td>
	      <td>Swin-B</td>
	      <td>COCO,O365,GoldG,Cap4M,OpenImage,ODinW-35,RefCOCO</td>
	      <td><a href="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth">Github link</a>  | <a href="https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth">HF link</a> 
	      <td><a href="https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/GroundingDINO_SwinB.cfg.py">link</a></td>
	    </tr>
	  </tbody>
	</table>
	
	
	
	
	
	
	2. set config file and args in utils/conf.py ,add your download weights to " MODEL_xxxx_PATH“  ,if need chatgpt,configure the "PROXIES", "API_KEY " 
	2. run demo
	
	```bash
	"--tag2text" :  provide images tage , you can use chatgpt to merge or filter words
	"--input_prompt" :  Select the detection target noun you are interested in, and you can turn off Tag2text
	'--color-flag': Give your semantic segmentation MASK the same category the same color
	```
	

```bash
python demo.py  --source <data path>  --save-txt  --save-mask --save-xml  --save_caption 
```

**🏃Demo **



![image-20230427093103453](asset/image-20230427093103453.png)



![image-20230508075845259](asset/webdemo.png)



***🔨To Do list***

- [x] Release demo and code(2 days within).
- [x] web ui   demo 
- [ ] support video ,chatgpt, add inpainting model demo
- [ ] add 3d nerf demo 
- [ ] fintune sam and ground?? 
- [ ] Release training datasets.

## 

## :cupid: Acknowledgements

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Tag2text](https://github.com/xinyu1205/Tag2Text) 
- [lama](https://github.com/advimman/lama) 

   Thanks for their great work!


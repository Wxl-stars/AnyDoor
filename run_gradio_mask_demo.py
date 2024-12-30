import cv2
import einops
import numpy as np
import torch
import random
import gradio as gr
import os
import albumentations as A
from PIL import Image
import torchvision.transforms as T
from datasets.data_utils import * 
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from omegaconf import OmegaConf
from cldm.hack import disable_verbosity, enable_sliced_attention
from PIL import Image


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()


config = OmegaConf.load('./configs/demo.yaml')
model_ckpt =  config.pretrained_model
model_config = config.config_file
use_interactive_seg = config.config_file

model = create_model(model_config ).cpu()
model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

if use_interactive_seg:
    from iseg.coarse_mask_refine_util import BaselineModel
    model_path = './iseg/coarse_mask_refine.pth'
    iseg_model = BaselineModel().eval()
    weights = torch.load(model_path , map_location='cpu')['state_dict']
    iseg_model.load_state_dict(weights, strict= True)

def process_choice(choice):
    return choice

def process_image_mask(image_np, mask_np):
    img = torch.from_numpy(image_np.transpose((2, 0, 1)))
    img = img.float().div(255).unsqueeze(0)
    mask = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
    pred = iseg_model(img, mask)['instances'][0,0].detach().numpy() > 0.5 
    return pred.astype(np.uint8)

def crop_back( pred, tar_image,  extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop    
    pred = cv2.resize(pred, (W2, H2))
    m = 3 # maigin_pixel

    if W1 == H1:
        tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]
    tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
    return tar_image




ref_dir='./hf_test/ref'
image_dir='./hf_test/BG'
ref_list=[os.path.join(ref_dir,file) for file in os.listdir(ref_dir) if '.jpg' in file or '.png' in file or '.jpeg' in file ]
ref_list.sort()
image_list=[os.path.join(image_dir,file) for file in os.listdir(image_dir) if '.jpg' in file or '.png' in file or '.jpeg' in file]
image_list.sort()

def mask_image(image, mask):
    blanc = np.ones_like(image) * 255
    mask = np.stack([mask,mask,mask],-1) / 255
    masked_image = mask * ( 0.5 * blanc + 0.5 * image) + (1-mask) * image
    return masked_image.astype(np.uint8)

def run_local(ref,
              label,
              *args):
    ref_image = ref["image"].convert("RGB")
    ref_mask = ref["mask"].convert("L")
    ref_image = np.asarray(ref_image)
    ref_mask = np.asarray(ref_mask)
    ref_mask = np.where(ref_mask > 128, 1, 0).astype(np.uint8)

    import time
    timestamp = str(int(time.time()))

    # save to disk
    cv2.imwrite(f"./hf_test/{label}_{timestamp}_ref_image.png", ref_image)
    cv2.imwrite(f"./hf_test/{label}_{timestamp}_ref_mask.png", ref_mask * 255)

    if ref_mask.sum() == 0:
        raise gr.Error('No mask for the reference image.')

    # if reference_mask_refine:
    ref_mask = process_image_mask(ref_image, ref_mask)
    cv2.imwrite(f"./hf_test/{label}_{timestamp}_refine_mask.png", ref_mask*255)

    ref_image_mask = ref_image * (np.stack([ref_mask, ref_mask, ref_mask], -1))
    cv2.imwrite(f"./hf_test/{label}_{timestamp}_image_mask.png", ref_image_mask[:, :, ::-1])
    # print(ref_image_mask.shape)
    # display = cv2.resize(ref_image_mask, (600, 300))
    return ref_image_mask



with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("#  Ref Object mask.")


        with gr.Row():
            ref = gr.Image(label="Reference", source="upload", tool="sketch", type="pil", height=1080, brush_color='#FFFFFF', mask_opacity=0.5)
        
        
        with gr.Row():
            with gr.Column():
                gr.Examples(ref_list, inputs=[ref],label="Examples - Reference Object",examples_per_page=16)
        
        # 创建下拉选项列表
        dropdown_options = ["cone", "water_horse", "construction_sign", "fence", "collision_bar", "speed_bump", "anti_collision_barrel"]
        label_input = gr.Dropdown(dropdown_options, label="请选择对应的label")

        run_local_button = gr.Button(label="save", value="Run")
        image_display = gr.Image(label="Displayed Image", height=512)  # 显示图像
        # baseline_gallery = gr.Gallery(label='Output', show_label=True, elem_id="gallery", columns=1, height=512, min_width=512)
        


    run_local_button.click(fn=run_local, 
                           inputs=[ref, label_input], 
                           outputs=[image_display]
                        )

demo.launch(server_name="0.0.0.0", share=True)

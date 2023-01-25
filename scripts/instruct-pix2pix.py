from __future__ import annotations

import math
import random
import sys
from argparse import ArgumentParser

import einops
import gradio as gr
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
import os
import shutil
import time
import stat
import gradio as gr
import modules.extras
from modules.ui_common import create_output_panel
import json
import re
import modules.images as images
from modules.shared import opts, cmd_opts
from modules import shared, scripts
from modules import script_callbacks
from pathlib import Path
from typing import List, Tuple
from PIL.ExifTags import TAGS
from PIL.PngImagePlugin import PngImageFile, PngInfo
from datetime import datetime
from modules.generation_parameters_copypaste import quote
import modules.generation_parameters_copypaste as parameters_copypaste
sys.path.append("../../")

## Uses example code from https://github.com/timothybrooks/instruct-pix2pix
## See accompanying license file for more info

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)

def generate(
    input_image: Image.Image,
    instruction: str,
    steps: int,
    randomize_seed: bool,
    seed: int,
    randomize_cfg: bool,
    text_cfg_scale: float,
    image_cfg_scale: float,
    ):
    #parser = ArgumentParser()
    #parser.add_argument("--resolution", default=512, type=int)
    #parser.add_argument("--config", default="configs/generate.yaml", type=str)
    #parser.add_argument("--ckpt", default="checkpoints/instruct-pix2pix-00-22000.ckpt", type=str)
    #parser.add_argument("--vae-ckpt", default=None, type=str)
    #args = parser.parse_args()

  
    #config = OmegaConf.load(args.config)
    model = shared.sd_model
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])
    #example_image = Image.open("imgs/example.jpg").convert("RGB")
    seed = random.randint(0, 100000) if randomize_seed else seed
    text_cfg_scale = round(random.uniform(6.0, 9.0), ndigits=2) if randomize_cfg else text_cfg_scale
    image_cfg_scale = round(random.uniform(1.2, 1.8), ndigits=2) if randomize_cfg else image_cfg_scale
    
    width, height = input_image.size
    factor = 512 / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

    if instruction == "":
        return [input_image, seed]

    with torch.no_grad(), autocast("cuda"), model.ema_scope():
        cond = {}
        cond["c_crossattn"] = [model.get_learned_conditioning([instruction])]
        input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
        cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

        uncond = {}
        uncond["c_crossattn"] = [null_token]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

        sigmas = model_wrap.get_sigmas(steps)

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": text_cfg_scale,
            "image_cfg_scale": image_cfg_scale,
        }
        torch.manual_seed(seed)
        z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        x = model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())

        generation_params = {
            "ip2p": "Yes",
            "Prompt:": instruction,
            "Steps": steps,
            "Sampler": "Euler A",
            "Image CFG scale": image_cfg_scale,
            "Text CFG scale": image_cfg_scale,
            "Seed": seed,
            "Model hash": (None if not opts.add_model_hash_to_info or not shared.sd_model.sd_model_hash else shared.sd_model.sd_model_hash),
            "Model": (None if not opts.add_model_name_to_info or not shared.sd_model.sd_checkpoint_info.model_name else shared.sd_model.sd_checkpoint_info.model_name.replace(',', '').replace(':', '')),
 
        }
        generation_params_text = ", ".join([k if k == v else f'{k}: {quote(v)}' for k, v in generation_params.items() if v is not None])
        
        images.save_image(Image.fromarray(x.type(torch.uint8).cpu().numpy()), "outputs/ip2p-images", "ip2p", seed, instruction, "png", info=generation_params_text)
        images_array = []
        images_array.append(edited_image)
        return [seed, text_cfg_scale, image_cfg_scale, images_array]

def reset():
    return [0, "Randomize Seed", 1371, "Fix CFG", 7.5, 1.5, None]


help_text = """
If you're not getting what you want, there may be a few reasons:
1. Is the image not changing enough? Your Image CFG weight may be too high. This value dictates how similar the output should be to the input. It's possible your edit requires larger changes from the original image, and your Image CFG weight isn't allowing that. Alternatively, your Text CFG weight may be too low. This value dictates how much to listen to the text instruction. The default Image CFG of 1.5 and Text CFG of 7.5 are a good starting point, but aren't necessarily optimal for each edit. Try:
    * Decreasing the Image CFG weight, or
    * Incerasing the Text CFG weight, or
2. Conversely, is the image changing too much, such that the details in the original image aren't preserved? Try:
    * Increasing the Image CFG weight, or
    * Decreasing the Text CFG weight
3. Try generating results with different random seeds by setting "Randomize Seed" and running generation multiple times. You can also try setting "Randomize CFG" to sample new Text CFG and Image CFG values each time.
4. Rephrasing the instruction sometimes improves results (e.g., "turn him into a dog" vs. "make him a dog" vs. "as a dog").
5. Increasing the number of steps sometimes improves results.
6. Do faces look weird? The Stable Diffusion autoencoder has a hard time with faces that are small in the image. Try:
    * Cropping the image so the face takes up a larger portion of the frame.
"""


example_instructions = [
    "Make it a picasso painting",
    "as if it were by modigliani",
    "convert to a bronze statue",
    "Turn it into an anime.",
    "have it look like a graphic novel",
    "make him gain weight",
    "what would he look like bald?",
    "Have him smile",
    "Put him in a cocktail party.",
    "move him at the beach.",
    "add dramatic lighting",
    "Convert to black and white",
    "What if it were snowing?",
    "Give him a leather jacket",
    "Turn him into a cyborg!",
    "make him wear a beanie",
]


def create_tab(tabname):
 
        with gr.Row(visible=True, elem_id="ip2p_tab") as main_panel:
    
            
            with gr.Column():
                with gr.Row():
                    with gr.Column(elem_id=f"ip2p_prompt_container", scale=6):
                        prompt = gr.Textbox(label="Prompt", elem_id=f"ip2p_prompt", show_label=False, lines=2, placeholder="Prompt (press Ctrl+Enter or Alt+Enter to generate)")
 
                    with gr.Row(elem_id=f"ip2p_generate_box"):
                        generate_button = gr.Button("Generate")
      
                with gr.Row():
                    input_image = gr.Image(label="Image for ip2p", elem_id="ip2p_image", show_label=False, source="upload", interactive=True, type="pil", tool="editor").style(height=480)
                    ip2p_gallery, html_info_x, html_info, html_log = create_output_panel("ip2p", "outputs/ip2p-images")
                #with gr.Column():
                        #ip2p_button = gr.Button("Back to input")

                with gr.Row():
                    steps = gr.Number(value=100, precision=0, label="Steps", interactive=True)
                    randomize_seed = gr.Radio(
                        ["Fix Seed", "Randomize Seed"],
                        value="Randomize Seed",
                        type="index",
                        show_label=False,
                        interactive=True,
                    )
                    seed = gr.Number(value=1371, precision=0, label="Seed", interactive=True)
                    randomize_cfg = gr.Radio(
                        ["Fix CFG", "Randomize CFG"],
                        value="Fix CFG",
                        type="index",
                        show_label=False,
                        interactive=True,
                    )
                    text_cfg_scale = gr.Number(value=7.5, label=f"Text CFG", interactive=True)
                    image_cfg_scale = gr.Number(value=1.5, label=f"Image CFG", interactive=True)

                    generate_button.click(
                        fn=generate,
                        inputs=[
                            input_image,
                            prompt,
                            steps,
                            randomize_seed,
                            seed,
                            randomize_cfg,
                            text_cfg_scale,
                            image_cfg_scale,
                        ],
                        outputs=[seed, text_cfg_scale, image_cfg_scale, ip2p_gallery],
                        show_progress=True,
                    )

                    #ip2p_button.click(fn=lambda *x: x, show_progress=False,inputs=[ip2p_gallery], outputs=[input_image])


tabs_list = ["ip2p"]
def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as i2p2p:
        with gr.Tabs(elem_id="ip2p_tab)") as tabs:
            for tab in tabs_list:
                with gr.Tab(tab):
                    with gr.Blocks(analytics_enabled=False):    
                        create_tab(tab)
    return (i2p2p, "Instruct-pix2pix", "ip2p"),

def on_ui_settings():
    section = ('ip2p', "Instruct-pix2pix")

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)

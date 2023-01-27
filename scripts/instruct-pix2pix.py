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
from modules.ui_components import FormRow, FormGroup, ToolButton, FormHTML
from modules.ui import create_toprow, process_interrogate, interrogate, interrogate_deepbooru, apply_styles, update_token_counter
import json
import re
import modules.images as images
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, wrap_gradio_call
from modules import ui_extra_networks
from modules.shared import opts, cmd_opts, OptionInfo
from modules import shared, scripts
from modules import script_callbacks
from pathlib import Path
from typing import List, Tuple
from PIL.ExifTags import TAGS
from PIL.PngImagePlugin import PngImageFile, PngInfo
from datetime import datetime
from modules.generation_parameters_copypaste import quote

from modules.sd_samplers import samplers, samplers_for_img2img
import modules.generation_parameters_copypaste as parameters_copypaste
sys.path.append("../../")
outdir = opts.outdir_samples
if opts.outdir_samples == "":
    shared.opts.add_option("outdir_ip2p_samples", shared.OptionInfo("outputs/ip2p-images", "Save path for images", section=('ip2p', "Instruct-pix2pix")))
    outdir = opts.outdir_ip2p_samples

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
    negative_prompt: str,
    batch_number: int,
    scale: int,
    batch_in_check,
    batch_in_dir
    ):


    input_images = []
    if (input_image is None and batch_in_check is False) or (input_image is None and (os.path.exists(batch_in_dir) == False or(batch_in_check and batch_in_dir == "")) ):
        return [seed, text_cfg_scale, image_cfg_scale, None]

    if batch_in_check and os.path.exists(batch_in_dir):
        for filename in os.listdir(batch_in_dir):
            with open(os.path.join(batch_in_dir, filename), 'rb') as f: # open in readonly mode
                try:
                    im=Image.open(f)
                    print(f"Adding image: " + filename)
                    input_images.append(filename)
                except IOError:
                    print(f"Ignoring non-image file: " + f)
        
    else:
        input_images.append(input_image)
        
        
    
   
    if instruction == "" and negative_prompt == "":
        return [input_image, seed]

    images_array = []
    orig_seed = seed

    orig_batch_number = batch_number

    print(f"Processing {len(input_images)} images")

    while batch_number > 0:
        while len(input_images) > 0:
  
            if batch_in_check:
                filename = input_images.pop(0)
                input_image = Image.open(os.path.join(batch_in_dir, filename))
            else:
                input_image = input_images.pop(0)
                            
            cur_batch_number = orig_batch_number

            model = shared.sd_model
            model.eval().cuda()
            model_wrap = K.external.CompVisDenoiser(model)
            model_wrap_cfg = CFGDenoiser(model_wrap)
            null_token = model.get_learned_conditioning([""])
            seed = random.randint(0, 100000) if randomize_seed else seed
            text_cfg_scale = round(random.uniform(6.0, 9.0), ndigits=2) if randomize_cfg else text_cfg_scale
            image_cfg_scale = round(random.uniform(1.2, 1.8), ndigits=2) if randomize_cfg else image_cfg_scale
            
            width, height = input_image.size
            factor = scale / max(width, height)
            factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
            width = int((width * factor) // 64) * 64
            height = int((height * factor) // 64) * 64
            input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

            with torch.no_grad(), autocast("cuda"), model.ema_scope():
               
                cond = {}
                cond["c_crossattn"] = [model.get_learned_conditioning([instruction])]
                input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
                input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
                cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

                uncond = {}
                uncond["c_crossattn"] = [model.get_learned_conditioning([negative_prompt])]
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
                    "Prompt": instruction,
                    "Negative Prompt": negative_prompt,
                    "Steps": steps,
                    "Sampler": "Euler A",
                    "Image CFG scale": image_cfg_scale,
                    "Text CFG scale": image_cfg_scale,
                    "Seed": seed,
                    "Model hash": (None if not opts.add_model_hash_to_info or not shared.sd_model.sd_model_hash else shared.sd_model.sd_model_hash),
                    "Model": (None if not opts.add_model_name_to_info or not shared.sd_model.sd_checkpoint_info.model_name else shared.sd_model.sd_checkpoint_info.model_name.replace(',', '').replace(':', '')),
         
                }
                generation_params_text = ", ".join([k if k == v else f'{k}: {quote(v)}' for k, v in generation_params.items() if v is not None])
            
                images.save_image(Image.fromarray(x.type(torch.uint8).cpu().numpy()), outdir, "ip2p", seed, instruction, "png", info=generation_params_text)
            
                images_array.append(edited_image)
                

        batch_number -= 1
        seed += 1
    return [orig_seed, text_cfg_scale, image_cfg_scale, images_array]

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


def add_style(name: str, prompt: str, negative_prompt: str):
    if name is None:
        return [gr_show() for x in range(4)]

    style = modules.styles.PromptStyle(name, prompt, negative_prompt)
    shared.prompt_styles.styles[style.name] = style
    # Save all loaded prompt styles: this allows us to update the storage format in the future more easily, because we
    # reserialize all styles every time we save them
    shared.prompt_styles.save_styles(shared.styles_filename)

    return [gr.Dropdown.update(visible=True, choices=list(shared.prompt_styles.styles)) for _ in range(2)]

def create_tab(tabname):
        
            
        with gr.Column(visible=True, elem_id="ip2p_tab") as main_panel:
            ip2p_prompt, ip2p_prompt_styles, ip2p_negative_prompt, submit, ip2p_interrogate, ip2p_deepbooru, ip2p_prompt_style_apply, ip2p_save_style, ip2p_paste, extra_networks_button, token_counter, token_button, negative_token_counter, negative_token_button = create_toprow(is_img2img=True)
##            with gr.Row():
##                with gr.Column(elem_id=f"ip2p_prompt_container", scale=6):
##                    prompt = gr.Textbox(label="Prompt", elem_id=f"ip2p_prompt", show_label=False, lines=2, placeholder="Prompt")
##                    negative_prompt = gr.Textbox(label="Negative Prompt", elem_id=f"ip2p_negative_prompt", show_label=False, lines=2, placeholder="Negative Prompt")
##                with gr.Column():
##                    generate_button = gr.Button(value="Generate", show_progress=True)
            with FormRow(variant='compact', elem_id="ip2p_extra_networks", visible=False) as extra_networks:
                from modules import ui_extra_networks
                extra_networks_ui_ip2p = ui_extra_networks.create_ui(extra_networks, extra_networks_button, 'ip2p')
            dummy_component = gr.Label(visible=False)
            with gr.Row():
                with gr.Tabs(elem_id="mode_ip2p"):
                    with gr.Row():
                        with gr.Tab(elem_id="input_ip2p", label="Input"):
                            with gr.Row():
                                with gr.Column():
                                    init_img = gr.Image(label="Disabled for batch input images", elem_id="ip2p_image", show_label=True, source="upload", type="pil")
                                    with gr.Row():
                                        batch_in_check = gr.Checkbox(label="Use batch input directory as image source")
                                        batch_in_dir = gr.Textbox(label="Directory for batch input images")
                                        batch_out_dir = gr.Textbox(label="Directory for batch output images", visible=False) 
                                    with gr.Row():
                                        batch_number = gr.Number(value=1, label="Output Batches", precision=0, interactive=True)
                                        steps = gr.Number(value=10, precision=0, label="Steps", interactive=True)
                                    with gr.Row():
                                        seed = gr.Number(value=1371, precision=0, label="Seed", interactive=True, show_progress=False)
                                        randomize_seed = gr.Radio(
                                            ["Fix Seed", "Randomize Seed"],
                                            value="Randomize Seed",
                                            type="index",
                                            show_label=False,
                                            interactive=True,
                                        )                                        
                                    with gr.Row():
                                        text_cfg_scale = gr.Number(value=7.5, precision=None, label=f"Text CFG", interactive=True, max_width=10, step=0.01, show_progress=False)
                                        image_cfg_scale = gr.Number(value=1.5, precision=None, label=f"Image CFG", interactive=True, max_width=10, step=0.01, show_progress=False)
                                        randomize_cfg = gr.Radio(  
                                            ["Fix CFG", "Randomize CFG"],
                                            value="Fix CFG",
                                            type="index",
                                            show_label=False,
                                            interactive=True,
                                        )
                                    with gr.Row(max_width=50):
                                         scale = gr.Slider(minimum=64, maximum=2048, step=8, label="Output Image Width", value=512, elem_id="ip2p_scale") 
                with gr.Tabs(elemn_id="output_ip2p"):
                    with gr.TabItem(elem_id="output_ip2p", label="Output"):
                        ip2p_gallery, html_info_x, html_info, html_log = create_output_panel("ip2p", outdir)
     

                    interrogate_args = dict(
                        _js="get_img2img_tab_index",
                        inputs=[
                            dummy_component,
                            batch_in_dir,
                            batch_in_dir,
                            init_img                            
                        ],
                        outputs=[ip2p_prompt, dummy_component],
                    )                   
                           
                    gen_inputs=[
                        init_img,
                        ip2p_prompt,
                        steps,
                        randomize_seed,
                        seed,
                        randomize_cfg,
                        text_cfg_scale,
                        image_cfg_scale,
                        ip2p_negative_prompt,
                        batch_number,
                        scale,
                        batch_in_check,
                        batch_in_dir
                    ]

                    gen_outputs=[
                        seed,
                        text_cfg_scale,
                        image_cfg_scale,
                        ip2p_gallery
                    ]

                    submit.click(
                        fn=generate,
                        inputs=gen_inputs,
                        outputs=gen_outputs,
                        show_progress=True,
                    )

                    ip2p_prompt.submit(
                        fn=generate,
                        inputs=gen_inputs,
                        outputs=gen_outputs,
                        show_progress=True,
                    )

                    ip2p_negative_prompt.submit(
                        fn=generate,
                        inputs=gen_inputs,
                        outputs=gen_outputs,
                        show_progress=True,
                    )

                    ip2p_interrogate.click(
                        fn=lambda *args: process_interrogate(interrogate, *args),
                        **interrogate_args,
                    )

                    ip2p_deepbooru.click(
                        fn=lambda *args: process_interrogate(interrogate_deepbooru, *args),
                        **interrogate_args,
                    )

                    prompts = [(ip2p_prompt, ip2p_negative_prompt)]
                    style_dropdowns = [ip2p_prompt_styles]
                    style_js_funcs = ["update_ip2p_tokens"]

            for button, (ip2p_prompt, ip2p_negative_prompt) in zip([ip2p_save_style], prompts):
                button.click(
                    fn=add_style,
                    _js="ask_for_style_name",
                    # Have to pass empty dummy component here, because the JavaScript and Python function have to accept
                    # the same number of parameters, but we only know the style-name after the JavaScript prompt
                    inputs=[dummy_component, ip2p_prompt, ip2p_negative_prompt],
                    outputs=[ip2p_prompt_styles, ip2p_prompt_styles],
                )

            for button, (ip2p_prompt, ip2p_negative_prompt), styles, js_func in zip([ip2p_prompt_style_apply], prompts, style_dropdowns, style_js_funcs):
                button.click(
                    fn=apply_styles,
                    #_js=js_func,
                    inputs=[ip2p_prompt, ip2p_negative_prompt, styles],
                    outputs=[ip2p_prompt, ip2p_negative_prompt, styles],
                )

            token_button.click(fn=update_token_counter, inputs=[ip2p_prompt, steps], outputs=[token_counter])
            negative_token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[ip2p_negative_prompt, steps], outputs=[negative_token_counter])

            ui_extra_networks.setup_ui(extra_networks_ui_ip2p, ip2p_gallery)

            ip2p_paste_fields = [
                (ip2p_prompt, "Prompt"),
                (ip2p_negative_prompt, "Negative prompt"),
                (steps, "Steps"),
                #(sampler_index, "Sampler"),
                #(restore_faces, "Face restoration"),
                (image_cfg_scale, "Image CFG scale"),
                (text_cfg_scale, "Text CFG scale"),
                (seed, "Seed"),
                (batch_number, "Batch size")
            ]
            parameters_copypaste.add_paste_fields("ip2p", init_img, ip2p_paste_fields)

                               
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

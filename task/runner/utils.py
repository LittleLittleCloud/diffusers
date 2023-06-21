import logging
import os

import torch
from diffusers.utils.loras import load_lora_weights
from diffusers import StableDiffusionPipeline
import omegaconf

def get_path(directory_path: str, file_path: str) -> str:
    if os.path.isabs(file_path):
        return file_path
    else:
        return os.path.join(directory_path, file_path)
    
def load_pipeline(
        cfg: omegaconf.omegaconf,
        directory_path: str,
        logger: logging) -> StableDiffusionPipeline:
    logger.info(f'Loading pipeline')
    base_model = cfg.base_model
    base_model = get_path(directory_path, base_model)
    logger.info(f'Loading base model from {base_model}')
    sampler = 'sampler' in cfg and cfg.sampler or 'ddim'
    device = 'device' in cfg and cfg.device or 'cuda'
    dtype = 'dtype' in cfg and cfg.dtype or torch.float16

    logger.info(f'Loading model from {base_model}')
    logger.info(f'Using sampler: {sampler}')
    pipe = StableDiffusionPipeline.from_ckpt(
        base_model,
        load_safety_checker = False,
        scheduler_type=sampler,
        torch_dtype=dtype)
    pipe = pipe.to(device)

    loras = 'loras' in cfg and cfg.loras or []
    for lora in loras:
        lora.name = get_path(directory_path, lora.name)
        logger.info(f'Applying lora: {lora.name} with weight {lora.weight}')
        pipe = load_lora_weights(
                pipeline = pipe,
                checkpoint_path=lora.name,
                multiplier=lora.weight,
                device=device,
                dtype=dtype)
    return pipe
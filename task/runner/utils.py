import logging
import os

import torch
from diffusers.utils.loras import load_lora_weights
from diffusers import StableDiffusionPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_sde import DPMSolverSDEScheduler
import omegaconf

def get_path(directory_path: str, file_path: str) -> str:
    if os.path.isabs(file_path):
        return file_path
    else:
        res = os.path.join(directory_path, file_path)
        if not os.path.exists(res):
            return file_path
        else:
            return res

def get_local_path(directory_path: str, file_path: str) -> str:
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
        # scheduler_type = 'ddim',
        torch_dtype=dtype)
    pipe = pipe.to(device)

    logger.info(pipe.scheduler.config)
    if sampler == 'ddim':
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif sampler == 'dpm':
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    elif sampler == 'dpm++ 2m':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif sampler == 'dpm++ 2m karras':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    else:
        raise NotImplementedError(f'Sampler {sampler} not implemented')

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
    
    def get_timesteps(num_inference_steps, strength = 0.8):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        print(pipe.scheduler.order)
        timesteps = pipe.scheduler.timesteps[t_start * pipe.scheduler.order :].to(dtype=torch.int32)

        return timesteps, num_inference_steps - t_start
    pipe.get_timesteps = get_timesteps
    
    return pipe

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
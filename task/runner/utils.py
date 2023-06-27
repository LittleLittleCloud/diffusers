import json
import logging
import os
import shutil

import torch
from diffusers.utils.loras import load_lora_weights
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_sde import DPMSolverSDEScheduler
import omegaconf
from diffusers.utils.torch_utils import randn_tensor
from task.config.dataset_config import *
from task.config.model_config import *
from datasets import load_dataset, Image, Dataset, load_from_disk
from task.log import get_logger
import glob

Logger = get_logger(__name__)

def get_path(cwd: str, file_path: str) -> str:
    if os.path.isabs(file_path):
        return file_path
    else:
        res = os.path.join(cwd, file_path)
        if not os.path.exists(res):
            return file_path
        else:
            return res

def get_local_path(directory_path: str, file_path: str) -> str:
    if os.path.isabs(file_path):
        return file_path
    else:
        return os.path.join(directory_path, file_path)

def load_stable_diffusion_img2img_pipeline(cwd: str, cfg: StableDiffusionModelConfig, device:str = 'cuda', dtype = torch.float16) -> StableDiffusionImg2ImgPipeline:
    Logger.info(f'Loading pipeline')
    base_model = cfg.base_model.model_name
    base_model = get_path(cwd, base_model)
    Logger.info(f'Loading base model from {base_model}')
    pipe = StableDiffusionImg2ImgPipeline.from_ckpt(
        base_model,
        load_safety_checker = False,
        torch_dtype=dtype)
    
    pipe = pipe.to(device)
    loras = cfg.loras
    for lora in loras:
        Logger.info(f'Applying lora: {lora.model} with weight {lora.weight}')
        model_name = get_path(cwd, lora.model.model_name)
        pipe = load_lora_weights(
                pipeline = pipe,
                checkpoint_path=model_name,
                multiplier=lora.weight,
                device=device,
                dtype=dtype)
    
    def get_timesteps(num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = pipe.scheduler.timesteps[t_start * pipe.scheduler.order :]
        # to int32
        timesteps = timesteps.to(torch.int32)
        return timesteps, num_inference_steps - t_start
    pipe.get_timesteps = get_timesteps

    return pipe

def load_stable_diffusion_pipeline(cwd: str, cfg: StableDiffusionModelConfig, device:str = 'cuda', dtype = torch.float16) -> StableDiffusionPipeline:
    Logger.info(f'Loading pipeline')
    base_model = cfg.base_model.model_name
    base_model = get_path(cwd, base_model)
    Logger.info(f'Loading base model from {base_model}')
    pipe = StableDiffusionPipeline.from_ckpt(
        base_model,
        load_safety_checker = False,
        torch_dtype=dtype)
    
    pipe = pipe.to(device)
    loras = cfg.loras
    for lora in loras:
        Logger.info(f'Applying lora: {lora.model} with weight {lora.weight}')
        model_name = get_path(cwd, lora.model.model_name)
        pipe = load_lora_weights(
                pipeline = pipe,
                checkpoint_path=model_name,
                multiplier=lora.weight,
                device=device,
                dtype=dtype)

    return pipe

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

def create_dataset_from_image_folder(
        cwd: str,
        cfg: ImageFolderConfig) -> Dataset:
    Logger.info(f'Creating dataset from image folder')
    image_folder = cfg.image_folder
    image_folder = get_path(cwd, image_folder)
    patterns = cfg.patterns
    is_recursive = cfg.recursive
    image_column = cfg.image_column
    image_path_column = cfg.image_path_column
    metadata = cfg.metadata_name
    Logger.info(f'Image folder: {image_folder}')
    Logger.info(f'Patterns: {patterns}')
    Logger.info(f'Is recursive: {is_recursive}')

    if metadata is not None:
        Logger.info(f'Loading metadata from {metadata}')
        dataset = Dataset.from_json(metadata).cast_column(image_column, Image())
        Logger.info(f'Loaded dataset with {len(dataset)} samples')
        return dataset

    image_paths = []
    dataset = None
    for pattern in patterns:
        image_paths.extend(glob.glob(os.path.join(image_folder, pattern), recursive=is_recursive))
        dataset = Dataset.from_dict({image_column: image_paths, image_path_column: image_paths}) \
            .cast_column(image_column, Image())
    Logger.info(f'Loaded dataset with {len(image_paths)} samples')

    return dataset

def save_dataset_as_image_folder(dataset: Dataset, cfg: ImageFolderConfig):
    output_folder = cfg.image_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_column = cfg.image_column
    image_path_column = cfg.image_path_column
    label_column = cfg.label_column
    new_image_names =[f'{i}-{os.path.basename(image_path)}' for i, image_path in enumerate(dataset[image_path_column])]
    new_image_paths = [os.path.join(output_folder, new_image_name) for new_image_name in new_image_names]

    for image_path, new_image_path in zip(dataset[image_path_column], new_image_paths):
        shutil.copy(image_path, new_image_path)
    
    if cfg.metadata_name is None:
        return
    
    metadata = {
        image_column: new_image_paths,
        image_path_column: new_image_paths,
    }

    if label_column is not None and label_column in dataset.column_names:
        metadata[label_column] = dataset[label_column]

    with open(os.path.join(output_folder, cfg.metadata_name), 'w') as f:
        json.dump(metadata, f)
        Logger.info(f'Saved metadata to {os.path.join(output_folder, cfg.metadata_name)}')

def create_dataset_from_dataset_config(cwd: str, cfg: DatasetConfig) -> Dataset:
    Logger.info(f'Creating dataset from dataset config')
    dataset_name = cfg.dataset_name
    dataset_name = get_path(cwd, dataset_name)
    dataset_config = cfg.dataset_config_name

    Logger.info(f'Dataset name: {dataset_name}')
    Logger.info(f'Dataset config: {dataset_config}')
    dataset: Dataset
    if os.path.exists(dataset_name):
        Logger.info(f'Loading dataset from disk: {dataset_name}')
        dataset = load_from_disk(dataset_name)
    else:
        dataset = load_dataset(dataset_name, dataset_config)
    Logger.info(f'Loaded dataset with {len(dataset)} samples')

    return dataset


import omegaconf
import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils.loras import load_lora_weights
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.utils import randn_tensor
from task.log import get_logger
from datasets import load_dataset, Image, Dataset
from torchvision import transforms
from diffusers.image_processor import VaeImageProcessor
import os
import PIL
from task.runner.runner import Runner
from task.config.txt2image_inference_config import *
from task.config.model_config import *
from task.runner.utils import *
Logger = get_logger(__name__)

class Txt2ImgInferenceRunner(Runner):
    name: str = 'txt2img_inference.v0'
    def execute(self, cwd: str, config: omegaconf.omegaconf):
        Logger.debug(f'pipeline directory: {cwd}')
        cfg: Txt2ImageInferenceConfig = omegaconf.OmegaConf.structured(Txt2ImageInferenceConfig)
        cfg = omegaconf.OmegaConf.merge(cfg, config)
        pipe = load_stable_diffusion_pipeline(cwd, cfg.model, cfg.device)
        Logger.info(pipe.scheduler.config)
        if cfg.sampler == 'ddim':
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        elif cfg.sampler == 'dpm':
            pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        elif cfg.sampler == 'dpm++ 2m':
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        elif cfg.sampler == 'dpm++ 2m karras':
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        else:
            raise NotImplementedError(f'Sampler {cfg.sampler} not implemented')
        
        # generate image
        width = cfg.width
        height = cfg.height
        seed = cfg.seed
        step = cfg.step
        prompt_column = cfg.prompt_column
        negative_prompt_column = cfg.negative_prompt_column
        dtype = torch.float16 if cfg.dtype == 'float16' else torch.float32

        cfg_value = cfg.cfg
        num_images_per_prompt = cfg.num_images_per_prompt
        Logger.info(f'Generating image with width: {width}')
        Logger.info(f'Generating image with height: {height}')
        Logger.info(f'Generating image with seed: {seed}')
        Logger.info(f'Generating image with step: {step}')
        Logger.info(f'Generating image with cfg: {cfg}')
        Logger.info(f'Generating image with num_images_per_prompt: {num_images_per_prompt}')
        
        dataset = None
        generator = torch.manual_seed(seed)
        if cfg.input.folder is not None:
            dataset = create_dataset_from_image_folder(cwd, cfg.input.folder)
        elif cfg.input.dataset is not None:
            dataset = create_dataset_from_dataset_config(cwd, cfg.input.dataset)
        elif cfg.input.prompt is not None:
            input = {
                prompt_column: [cfg.input.prompt],
                negative_prompt_column: [cfg.input.negative_prompt],
            }
            dataset = Dataset.from_dict(input)
        else:
            raise ValueError('Input is not valid')
        
        output = cfg.output
        output_folder = get_local_path(cwd, output.image_folder)
        for i, row in enumerate(dataset):
            prompt = row[prompt_column]
            negative_prompt = row[negative_prompt_column]
            Logger.info(f'Generating image with prompt: {prompt}')
            Logger.info(f'Generating image with negative prompt: {negative_prompt}')
            images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                generator=generator,
                num_inference_steps=step,
                guidance_scale = cfg_value,
                num_images_per_prompt=num_images_per_prompt)
            with open(os.path.join(output_folder, f'{i}.txt'), 'w') as f:
                f.writelines([prompt, negative_prompt])
            for j, image in enumerate(images.images):
                image.save(os.path.join(output_folder, f'{i}-{j}.png'))
        
        del pipe
                
        
        
        
        
        



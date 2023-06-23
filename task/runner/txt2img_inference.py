import omegaconf
import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils.loras import load_lora_weights
from diffusers.utils import randn_tensor
from task.log import get_logger
from datasets import load_dataset, Image, Dataset
from torchvision import transforms
from diffusers.image_processor import VaeImageProcessor
import os
import PIL
from task.runner.runner import Runner
from task.runner.utils import load_pipeline, get_local_path
Logger = get_logger(__name__)

class Txt2ImgInferenceRunner(Runner):
    name: str = 'txt2img_inference.v0'
    def execute(self, config: omegaconf.omegaconf):
        pipeline_directory = os.path.dirname(config._pipeline_path)
        Logger.debug(f'pipeline directory: {pipeline_directory}')
        pipe = load_pipeline(config.model, pipeline_directory, Logger)
        # generate image
        width = 'width' in config and config.width or 512
        height = 'height' in config and config.height or 768
        seed = 'seed' in config and config.seed or -1
        step = 'step' in config and config.step or 15
        output = 'output' in config and config.output or {}
        output_folder = 'output_folder' in output and output.output_folder or 'output'
        cfg = 'cfg' in config and config.cfg or 7.0
        dtype = 'dtype' in config and config.dtype or torch.float16
        num_images_per_prompt = 'num_images_per_prompt' in config and config.num_images_per_prompt or 1
        input = 'input' in config and config.input or {}
        Logger.info(f'Generating image with width: {width}')
        Logger.info(f'Generating image with height: {height}')
        Logger.info(f'Generating image with seed: {seed}')
        Logger.info(f'Generating image with step: {step}')
        Logger.info(f'Generating image with cfg: {cfg}')
        Logger.info(f'Generating image with num_images_per_prompt: {num_images_per_prompt}')
        
        image_processor = VaeImageProcessor(vae_scale_factor=2 ** (len(pipe.vae.config.block_out_channels) - 1))
        dataset = None
        generator = torch.manual_seed(seed)
        if 'dataset' in input:
            pass

        elif 'prompt' in input:
            prompt = 'prompt' in input and input.prompt or ''
            negative_prompt = 'negative_prompt' in input and input.negative_prompt or ''
            image = 'image' in input and input.image or None
            strength = 'strength' in input and input.strength or 0.8
            if image is not None:
                image = get_local_path(pipeline_directory, image)
                Logger.info(f'Loading image from {image}')
                image = PIL.Image.open(image)
            dataset = Dataset.from_dict({'prompt': [prompt], 'negative_prompt': [negative_prompt], 'image': [image], 'strength': [strength]})
        if not os.path.isabs(output_folder):
            output_folder = os.path.join(pipeline_directory, output_folder)
        Logger.info(f'Saving images to {output_folder}')
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        for i, row in enumerate(dataset):
            prompt = row['prompt']
            negative_prompt = row['negative_prompt']
            image = row['image']
            strength = row['strength']
            latent = None
            if image is not None:
                image = image_processor.preprocess(image, width=width, height=height).to(device = pipe.device, dtype=dtype)
                # stack image
                timesteps, step = pipe.get_timesteps(step, strength)
                latent_timestep = timesteps[:1].repeat(num_images_per_prompt)

                init_latents = pipe.vae.encode(image).latent_dist.sample(generator)
                init_latents = init_latents * pipe.vae.config.scaling_factor
                init_latents = init_latents.repeat(num_images_per_prompt, 1, 1, 1)
                init_latents = init_latents.to(dtype=dtype)
                noise = randn_tensor(init_latents.shape, generator=generator, device=pipe.device, dtype=dtype)
                latent = pipe.scheduler.add_noise(init_latents, noise, latent_timestep)
            Logger.info(f'Generating image with prompt: {prompt}')
            Logger.info(f'Generating image with negative prompt: {negative_prompt}')
            images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                latents=latent,
                generator=generator,
                num_inference_steps=step,
                guidance_scale = cfg,
                num_images_per_prompt=num_images_per_prompt)
            with open(os.path.join(output_folder, f'{i}.txt'), 'w') as f:
                f.writelines([prompt, negative_prompt])
            for j, image in enumerate(images.images):
                image.save(os.path.join(output_folder, f'{i}-{j}.png'))
                
        
        
        
        
        



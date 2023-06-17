import omegaconf
import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils.loras import load_lora_weights
from task.log import get_logger
import os
Logger = get_logger(__name__)
class Runner:
    name: str = None
    def execute(self, config: omegaconf.omegaconf):
        pass

class Txt2ImgInferenceRunner(Runner):
    name: str = 'txt2img_inference.v0'
    def execute(self, config: omegaconf.omegaconf):
        pipeline_directory = os.path.dirname(config._pipeline_path)
        Logger.debug(f'pipeline directory: {pipeline_directory}')
        base_model = config.base_model
        Logger.debug(os.path.join(pipeline_directory, base_model))
        if os.path.exists(os.path.join(pipeline_directory, base_model)):
            base_model = os.path.join(pipeline_directory, base_model)
        sampler = config.sampler
        device = 'device' in config and config.device or 'cuda'
        dtype = 'dtype' in config and config.dtype or torch.float16

        Logger.info(f'Loading model from {base_model}')
        Logger.info(f'Using sampler: {sampler}')
        pipe = StableDiffusionPipeline.from_ckpt(
            base_model,
            load_safety_checker = False,
            scheduler_type=sampler,
            torch_dtype=dtype)
        pipe = pipe.to(device)

        loras = 'loras' in config and config.loras or []
        for lora in loras:
            if os.path.exists(os.path.join(pipeline_directory, lora.name)):
                lora.name = os.path.join(pipeline_directory, lora.name)
            Logger.info(f'Applying lora: {lora.name} with weight {lora.weight}')
            pipe = load_lora_weights(
                    pipeline = pipe,
                    checkpoint_path=lora.name,
                    multiplier=lora.weight,
                    device=device,
                    dtype=dtype)
        
        # generate image
        width = 'width' in config and config.width or 512
        height = 'height' in config and config.height or 768
        seed = 'seed' in config and config.seed or -1
        prompt = 'prompt' in config and config.prompt or ''
        negative_prompt = 'negative_prompt' in config and config.negative_prompt or ''
        step = 'step' in config and config.step or 15
        output = 'output' in config and config.output_folder or 'output'
        cfg = 'cfg' in config and config.cfg or 7.0
        num_images_per_prompt = 'num_images_per_prompt' in config and config.num_images_per_prompt or 1
        Logger.info(f'Generating image with prompt: {prompt}')
        Logger.info(f'Generating image with negative prompt: {negative_prompt}')
        Logger.info(f'Generating image with width: {width}')
        Logger.info(f'Generating image with height: {height}')
        Logger.info(f'Generating image with seed: {seed}')
        Logger.info(f'Generating image with step: {step}')
        Logger.info(f'Generating image with cfg: {cfg}')
        Logger.info(f'Generating image with num_images_per_prompt: {num_images_per_prompt}')

        images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            generator=torch.manual_seed(seed),
            num_inference_steps=step,
            guidance_scale = cfg,
            num_images_per_prompt=num_images_per_prompt)
        
        if not os.path.isabs(output):
            output = os.path.join(pipeline_directory, output)
        Logger.info(f'Saving images to {output}')
        
        if not os.path.exists(output):
            os.mkdir(output)
        
        for i, image in enumerate(images.images):
            image.save(os.path.join(output, f'{i}.png'))
            with open(os.path.join(output, f'{i}.txt'), 'w') as f:
                f.write(prompt)


        

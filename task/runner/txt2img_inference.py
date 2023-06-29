import omegaconf
import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils.loras import load_lora_weights
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.utils import randn_tensor
from task.log import get_logger
from datasets import load_dataset, Dataset
from torchvision import transforms
import PIL.Image
from diffusers.image_processor import VaeImageProcessor
import os
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
        dtype = torch.float16 if cfg.dtype == 'float16' else torch.float32

        mode: str
        pipe: Union[StableDiffusionPipeline, StableDiffusionImg2ImgPipeline] = None
        if  cfg.image_column is None:
            Logger.info("txt2img mode")
            mode = 'txt2img'
            pipe = load_stable_diffusion_pipeline(cwd, cfg.model, cfg.device, dtype)
        elif cfg.image_column is not None and cfg.inner_image_column is None:
            Logger.info("img2img mode")
            mode = 'img2img'
            pipe = load_stable_diffusion_img2img_pipeline(cwd, cfg.model, cfg.device)
        else:
            raise Exception("Invalid mode")

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
        image_column = cfg.image_column
        inner_image_column = cfg.inner_image_column
        strength = cfg.strength
        cfg_value = cfg.cfg
        num_images_per_prompt = cfg.num_images_per_prompt

        Logger.info(f'Generating image with width: {width}')
        Logger.info(f'Generating image with height: {height}')
        Logger.info(f'Generating image with seed: {seed}')
        Logger.info(f'Generating image with step: {step}')
        Logger.info(f'Generating image with cfg: {cfg}')
        Logger.info(f'Generating image with num_images_per_prompt: {num_images_per_prompt}')
        trans = transforms.Compose([
            transforms.Resize([max(width, height)]),
        ])
        dataset = None
        generator = torch.manual_seed(seed)
        if cfg.input.folder is not None:
            dataset = create_dataset_from_image_folder(cwd, cfg.input.folder)
        elif cfg.input.dataset is not None:
            dataset = create_dataset_from_dataset_config(cwd, cfg.input.dataset)
        elif cfg.input.prompt is not None:
            input = {
                prompt_column: [cfg.input.prompt],
            }
            if cfg.input.negative_prompt is not None:
                input[negative_prompt_column] = [cfg.input.negative_prompt]
            if mode == 'img2img':
                image_path = get_local_path(cwd, cfg.input.image)
                image = PIL.Image.open(image_path)
                # resize image to width and height
                image = trans(image)
                input[image_column] = [image]
            print(input)
            dataset = Dataset.from_dict(input)
        else:
            raise ValueError('Input is not valid')
        
        output = cfg.output
        output_folder = get_local_path(cwd, output.image_folder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_metadata = []
        for i, row in enumerate(dataset):
            prompt = row[prompt_column]
            negative_prompt = row[negative_prompt_column] if negative_prompt_column is not None else None

            # overrite prompt and negative prompt if provided
            if cfg.input.prompt is not None:
                Logger.info(f'Overwriting prompt with {cfg.input.prompt}')
                prompt = cfg.input.prompt
            if cfg.input.negative_prompt is not None:
                Logger.info(f'Overwriting negative prompt with {cfg.input.negative_prompt}')
                negative_prompt = cfg.input.negative_prompt
            
            args = {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'generator': generator,
                'num_inference_steps': step,
                'guidance_scale': cfg_value,
                'num_images_per_prompt': num_images_per_prompt
            }

            if mode == 'txt2img':
                args['width'] = width
                args['height'] = height
            if mode == 'img2img':
                image = row[image_column]
                image = trans(image)
                image_path = os.path.join(output_folder, f'{i}-input.png')
                image.save(image_path)
                args['image'] = [image] * num_images_per_prompt
                args['strength'] = strength
            Logger.info(f'Generating image with prompt: {prompt}')
            Logger.info(f'Generating image with negative prompt: {negative_prompt}')
            images = pipe(**args)
            with open(os.path.join(output_folder, f'{i}-args.json'), 'w') as f:
                del args['generator']
                if 'image' in args:
                    image_path = os.path.join(output_folder, f'{i}-input.png')
                    args['image'] = image_path
                f.write(json.dumps(args, indent=4))
            for j, image in enumerate(images.images):
                image_input_path = os.path.join(output_folder, f'{i}-input.png')
                image_path = os.path.join(output_folder, f'{i}-{j}.png')
                image.save(image_path)
                row = {
                    prompt_column: prompt,
                    cfg.output_image_column: image_path,
                }
                if negative_prompt_column is not None:
                    row[negative_prompt_column] = negative_prompt
                
                if mode == 'img2img':
                    row[image_column] = image_input_path
                output_metadata.append(row)
        
        output_metadata_path = os.path.join(output_folder, 'metadata.json')
        with open(output_metadata_path, 'w') as f:
            f.write(json.dumps(output_metadata, indent=4))
        del pipe
                
        
        
        
        
        



import omegaconf
import torch
import os
from task.runner.runner import Runner
from task.runner.utils import get_path, get_local_path
from task.log import get_logger
import transformers
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from datasets import load_dataset, Image, Dataset
import glob
import torchvision.transforms as transforms
import shutil
Logger = get_logger(__name__)

class BlipImageCaptionRunner(Runner):
    name: str = 'blip_image_caption.v0'
    def __call__(self,
                 pipeline_directory: str,
                 input_config: omegaconf.omegaconf,
                 model_config: omegaconf.omegaconf,
                 inference_config: omegaconf.omegaconf,
                 output_config: omegaconf.omegaconf) -> Dataset:
        Logger.debug(f'pipeline directory: {pipeline_directory}')
        dataset = None
        input_folder = 'input_folder' in input_config and input_config.input_folder or None
        image_paths = None
        if input_folder is not None:
            input_folder = get_path(pipeline_directory, input_folder)
            patterns = 'patterns' in input_config and input_config.patterns or ['**/*']
            recursive = 'recursive' in input_config and input_config.recursive or False
            Logger.info(f'Loading dataset from {input_folder}')
            Logger.info(f'patterns: {patterns}')
            Logger.info(f'recursive: {recursive}')
            image_paths = []
            for pattern in patterns:
                image_paths.extend(glob.glob(os.path.join(input_folder, pattern), recursive=recursive))
            dataset = Dataset.from_dict({"image": image_paths}).cast_column("image", Image())
            Logger.info(f'Loaded dataset with {len(image_paths)} samples')
        
        def compose(x):
            tensor = [image for image in x["image"]]
            x["image"] = tensor
            return x
            
        train_images = dataset.with_transform(compose)
        batch_size = 'batch_size' in inference_config and inference_config.batch_size or 1
        def batch(iterable, n=1):
            l = len(iterable)
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]
        output_texts = []
        checkpoint = model_config.checkpoint
        checkpoint = get_path(pipeline_directory, checkpoint)
        Logger.info(f'Loading checkpoint from {checkpoint}')
        model = Blip2ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.float16)
        processor = Blip2Processor.from_pretrained(checkpoint)
        trigger_world = 'trigger_world' in inference_config and inference_config.trigger_world or input_folder
        device = 'device' in inference_config and inference_config.device or 'cuda'
        model = model.to(device)
        for i, images in enumerate(batch(train_images['image'], batch_size)):
            Logger.info(f'Processing batch {i}')
            with torch.no_grad():
                inputs = processor(images, return_tensors="pt").to(device, torch.float16)
                outputs = model.generate(**inputs, max_new_tokens = 74)
                texts = processor.batch_decode(outputs, skip_special_tokens=True)
                print(texts)
                output_texts.extend([f'{t.rstrip()} ({trigger_world}) ({i * batch_size + j}-th-picture)' for j, t in enumerate(texts)])
        Logger.info(f'Finished processing {len(output_texts)} samples')
        image_column = 'image_column' in output_config and output_config.image_column or 'image'
        caption_column = 'caption_column' in output_config and output_config.caption_column or 'caption'
        dataset = dataset.add_column(caption_column, output_texts)
        if(image_column != 'image'):
            dataset.rename_column('image', image_column)
        output_folder = output_config.output_folder
        output_folder = get_local_path(pipeline_directory, output_folder)
        Logger.info(f'Writing output to {output_folder}')
        os.makedirs(output_folder, exist_ok=True)
        dataset.save_to_disk(output_folder)

        output_as_image_folder = 'output_as_image_folder' in output_config and output_config.output_as_image_folder or False
        if output_as_image_folder:
            Logger.info(f'output as image folder')
            Logger.info(f'Writing metadata to {output_folder}')
            with open(os.path.join(output_folder, 'metadata.csv'), 'w') as f:
                f.write(f'\"{image_column}\",\"{caption_column}\"\n')
                for i, image_path in enumerate(image_paths):
                    image_name = f'{i}-{os.path.basename(image_path)}'
                    f.write(f'\"{image_name},{output_texts[i]}\n')
                    # copy image to output folder
                    Logger.info(f'Writing image {image_name}')
                    shutil.copy(image_path, os.path.join(output_folder, image_name))
        Logger.info(f'Finished writing output to {output_folder}')
        del model
        return dataset

    def execute(self, config: omegaconf.omegaconf):
        pipeline_directory = os.path.dirname(config._pipeline_path)
        self.__call__(pipeline_directory, config.input, config.model, config.inference, config.output)

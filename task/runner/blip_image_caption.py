import omegaconf
import torch
import os
from task.runner.runner import Runner
from task.runner.utils import *
from task.log import get_logger
from task.config.image_caption_config import *
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
                 cwd: str,
                 cfg: BlipImageCaptionConfig) -> Dataset:
        Logger.info(f'pipeline directory: {cwd}')
        dataset = None
        input: Union[ImageFolderConfig, DatasetConfig] = None
        if cfg.input.folder is not None:
            input = cfg.input.folder
            input.image_folder = get_local_path(cwd, input.image_folder)
            dataset = create_dataset_from_image_folder(input)
        elif cfg.input.dataset is not None:
            input = cfg.input.folder
            input.image_folder = get_path(cwd, input.image_folder)
            dataset = create_dataset_from_dataset_config(input)
        else:
            raise Exception(f'input {input} not recognized')
        
        image_column = cfg.image_column
        caption_column = cfg.caption_column
        
        def compose(x):
            tensor = [image for image in x[image_column]]
            x[image_column] = tensor
            return x
            
        train_images = dataset.with_transform(compose)
        batch_size = cfg.batch_size
        output_texts = []
        model_config = cfg.model
        checkpoint = None
        if model_config is not None:
            checkpoint = get_path(cwd, model_config.model_name)
        else:
            raise Exception(f'model {model_config} not recognized')
        Logger.info(f'Loading model from {checkpoint}')
        model = Blip2ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.float16)
        processor = Blip2Processor.from_pretrained(checkpoint)
        trigger_world = cfg.trigger_word
        device = cfg.device
        model = model.to(device)
        for i, images in enumerate(batch(train_images[image_column], batch_size)):
            Logger.info(f'Processing batch {i}')
            with torch.no_grad():
                inputs = processor(images, return_tensors="pt").to(device, torch.float16)
                outputs = model.generate(**inputs, max_new_tokens = 74)
                texts = processor.batch_decode(outputs, skip_special_tokens=True)
                print(texts)
                output_texts.extend([f'{t.rstrip()} {trigger_world}' for j, t in enumerate(texts)])
        Logger.info(f'Finished processing {len(output_texts)} samples')
        if caption_column in dataset.column_names:
            Logger.info(f'Removing column {caption_column}')
            dataset = dataset.rename_column(caption_column)
        dataset = dataset.add_column(caption_column, output_texts)
        
        output = cfg.output
        if cfg.output.folder is not None:
            output.folder.image_folder = get_local_path(cwd, output.folder.image_folder)
            output.folder.image_column = image_column
            output.folder.label_column = caption_column
            save_dataset_as_image_folder(dataset, output.folder)

        del model
        return dataset

    def execute(self, cwd: str, config: omegaconf.omegaconf):
        cfg: BlipImageCaptionConfig = omegaconf.OmegaConf.structured(BlipImageCaptionConfig)
        cfg = omegaconf.OmegaConf.merge(cfg, config)
        self.__call__(cwd, cfg)

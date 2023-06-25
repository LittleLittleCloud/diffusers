from typing import Any
import omegaconf
import torch
import os
from task.config.image_caption_config import *
from task.runner.runner import Runner
from task.runner.tagger.interrogator import Interrogator, WaifuDiffusionInterrogator
from task.runner.utils import *
from task.log import get_logger
import transformers
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from datasets import load_dataset, Image, Dataset
import datasets
import glob
import torchvision.transforms as transforms
import shutil
Logger = get_logger(__name__)

class WaifuImageTaggerRunner(Runner):
    name: str = 'waifu_image_tagger.v0'
    def __call__(self,
                 cwd: str,
                 cfg: WDV1_4TaggerConfig) -> Dataset:
        Logger.debug(f'pipeline directory: {cwd}')
        dataset = None
        image_column = cfg.image_column
        caption_column = cfg.caption_column
        input: Union[ImageFolderConfig, DatasetConfig] = None
        if cfg.input.dataset is not None:
            input = cfg.input.dataset
            input.dataset_name = get_local_path(cwd, input.dataset_name)
            dataset = create_dataset_from_dataset_config(input)
        elif cfg.input.folder is not None:
            input = cfg.input.folder
            input.image_folder = get_path(cwd, input.image_folder)
            dataset = create_dataset_from_image_folder(input)
        else:
            raise Exception(f'input {input} not recognized')
        
        def compose(x):
            tensor = [image for image in x["image"]]
            x["image"] = tensor
            return x
            
        train_images = dataset.with_transform(compose)
        batch_size = cfg.batch_size
        output_texts = []
        model_config = cfg.model
        if model_config is not None:
            checkpoint = get_path(cwd, model_config.model_name)
        else:
            raise Exception(f'model {model_config} not recognized')
        Logger.info(f'Loading model from {checkpoint}')
        model = WaifuDiffusionInterrogator(checkpoint)
        for i, images in enumerate(batch(train_images['image'], batch_size)):
            Logger.info(f'Processing batch {i}')
            reses = model.interrogate(images)
            for res in reses:
                tags = res[1]
                filter_tags = [k for k, v in tags.items() if v >= cfg.filter_threshold]
                if len(filter_tags) == 0:
                    continue
                Logger.info(f'Found {len(filter_tags)} tags')
                Logger.info(f'{",".join(filter_tags)}')
                output_texts.append(f'{",".join(filter_tags)}')
        Logger.info(f'Finished processing {len(output_texts)} samples')
        if caption_column in dataset.column_names:
            Logger.info(f'Removing column {caption_column}')
            dataset = dataset.remove_column(caption_column)
        dataset = dataset.add_column(caption_column, output_texts)
        if cfg.output.folder is not None:
            cfg.output.folder.image_folder = get_local_path(cwd, cfg.output.folder.image_folder)
            cfg.output.folder.image_column = image_column
            cfg.output.folder.label_column = caption_column
            save_dataset_as_image_folder(dataset, cfg.output.folder)

        del model
        return dataset

    def execute(self, cwd: str, config: omegaconf.omegaconf):
        cfg: WDV1_4TaggerConfig = omegaconf.OmegaConf.structured(WDV1_4TaggerConfig)
        cfg = omegaconf.OmegaConf.merge(cfg, config)
        self.__call__(cwd, cfg)

import omegaconf
import torch
import os
from task.runner.runner import Runner
from task.runner.blip_image_caption import BlipImageCaptionRunner
from task.runner.tagger.waifu_tagger import WaifuImageTaggerRunner
from task.runner.utils import *
from task.config.image_caption_config import *
from task.log import get_logger
import transformers
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from datasets import load_dataset, Image, Dataset
import glob
import torchvision.transforms as transforms
import shutil
import copy
Logger = get_logger(__name__)

class ImageCaptionTaggerRunner(Runner):
    name: str = 'image_caption_tagger.v0'

    def execute(self, cwd: str, config: omegaconf.omegaconf):
        cfg: ImageCaptionTaggerConfig = omegaconf.OmegaConf.structured(ImageCaptionTaggerConfig)
        cfg = omegaconf.OmegaConf.merge(cfg, config)
        Logger.debug(f'pipeline directory: {cwd}')
        dataset = None
        image_column = cfg.image_column
        caption_column = cfg.caption_column

        if cfg.caption is not None:
            cfg.caption.output = copy.deepcopy(cfg.output)
            cfg.caption.input = copy.deepcopy(cfg.input)
            runner = BlipImageCaptionRunner()
            caption_dataset = runner(cwd, cfg.caption)
            if dataset is None:
                dataset = caption_dataset
            else:
                dataset = dataset.add_column(cfg.caption.caption_column, caption_dataset[cfg.caption.caption_column])
        
        if cfg.tagger is not None:
            cfg.tagger.output = copy.deepcopy(cfg.output)
            cfg.tagger.input = copy.deepcopy(cfg.input)
            runner = WaifuImageTaggerRunner()
            tagger_dataset = runner(cwd, cfg.tagger)
            if dataset is None:
                dataset = tagger_dataset
            else:
                dataset = dataset.add_column(cfg.tagger.caption_column, tagger_dataset[cfg.tagger.caption_column])
        
        # append caption and tagger
        def combine_caption_tagger(x):
            caption = x[cfg.caption.caption_column] if cfg.caption.caption_column in x else ''
            tagger = x[cfg.tagger.caption_column] if cfg.tagger.caption_column in x else ''
            x[caption_column] = f'{caption}, {tagger}, {cfg.trigger_word}'
            return x
        dataset = dataset.map(combine_caption_tagger)
        if cfg.caption.caption_column in dataset:
            Logger.info(f'Removing {cfg.caption.caption_column} column')
            dataset = dataset.remove_columns([cfg.caption.caption_column])
        if cfg.tagger.caption_column in dataset:
            Logger.info(f'Removing {cfg.tagger.caption_column} column')
            dataset = dataset.remove_columns([cfg.tagger.caption_column])

        if cfg.output.folder is not None:
            output = cfg.output.folder
            output.image_folder = get_local_path(cwd, output.image_folder)
            output.image_column = image_column
            output.label_column = caption_column
            save_dataset_as_image_folder(dataset, output)
            
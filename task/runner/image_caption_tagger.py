import omegaconf
import torch
import os
from task.runner.runner import Runner
from task.runner.blip_image_caption import BlipImageCaptionRunner
from task.runner.tagger.waifu_tagger import WaifuImageTaggerRunner
from task.runner.utils import get_path, get_local_path
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

    def execute(self, config: omegaconf.omegaconf):
        input_config = config.input
        output_config = config.output
        image_column = output_config.image_column
        caption_column = output_config.caption_column
        pipeline_directory = os.path.dirname(config._pipeline_path)
        dataset = None
        if 'caption_config' in config and config.caption_config != None:
            Logger.info(f'Running captioning')
            caption_output_config = copy.deepcopy(output_config)
            caption_output_config['output_folder'] = os.path.join(output_config['output_folder'], 'caption')
            caption_output_config['caption_column'] = '_caption'
            runner = BlipImageCaptionRunner()
            caption_dataset = runner.__call__(
                pipeline_directory=pipeline_directory,
                input_config = input_config,
                model_config = config.caption_config.model,
                inference_config = config.caption_config.inference,
                output_config = caption_output_config)

            if dataset is None:
                dataset = caption_dataset
            else:
                dataset['_caption'] = caption_dataset[caption_column]
        
        if 'tagger_config' in config and config.tagger_config != None:
            Logger.info(f'Running tagging')
            tagger_output_config = copy.deepcopy(output_config)
            tagger_output_config['output_folder'] = os.path.join(output_config['output_folder'], 'tagger')
            tagger_output_config['caption_column'] = '_tagger'
            runner = WaifuImageTaggerRunner()
            tagger_dataset = runner.__call__(
                pipeline_directory=pipeline_directory,
                input = input_config,
                model_config = config.tagger_config.model,
                inference = config.tagger_config.inference,
                output = tagger_output_config)
            if dataset is None:
                dataset = tagger_dataset
            else:
                dataset['_tagger'] = tagger_dataset[caption_column]
        
        # append caption and tagger
        def combine_caption_tagger(x):
            caption = x['_caption'] if '_caption' in x else ''
            tagger = x['_tagger'] if '_tagger' in x else ''
            x[caption_column] = f'{caption}, {tagger}'
            return x
        dataset = dataset.map(combine_caption_tagger)
        dataset = dataset.remove_columns(['_caption', '_tagger'])
        output_folder = output_config.output_folder
        output_folder = get_local_path(pipeline_directory, output_folder)
        Logger.info(f'Saving dataset to {output_folder}')
        os.makedirs(output_folder, exist_ok=True)
        dataset.save_to_disk(output_folder)

        input_folder = input_config.input_folder
        input_folder = get_local_path(pipeline_directory, input_folder)
        patterns = 'patterns' in input_config and input_config.patterns or ['**/*']
        recursive = 'recursive' in input_config and input_config.recursive or False
        Logger.info(f'Loading dataset from {input_folder}')
        Logger.info(f'patterns: {patterns}')
        Logger.info(f'recursive: {recursive}')
        image_paths = []
        for pattern in patterns:
            image_paths.extend(glob.glob(os.path.join(input_folder, pattern), recursive=recursive))
        output_as_image_folder = 'output_as_image_folder' in output_config and output_config.output_as_image_folder or False
        if output_as_image_folder:
            Logger.info(f'output as image folder')
            Logger.info(f'Writing metadata to {output_folder}')
            output_texts = dataset[caption_column]
            with open(os.path.join(output_folder, 'metadata.csv'), 'w') as f:
                f.write(f'\"{image_column}\",\"{caption_column}\"\n')
                for i, image_path in enumerate(image_paths):
                    image_name = f'{i}-{os.path.basename(image_path)}'
                    f.write(f'\"{image_name},{output_texts[i]}\n')
                    # copy image to output folder
                    Logger.info(f'Writing image {image_name}')
                    shutil.copy(image_path, os.path.join(output_folder, image_name))
        Logger.info(f'Finished writing output to {output_folder}')

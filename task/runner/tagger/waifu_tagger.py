from typing import Any
import omegaconf
import torch
import os
from task.runner.runner import Runner
from task.runner.tagger.interrogator import Interrogator, WaifuDiffusionInterrogator
from task.runner.utils import get_path, get_local_path, batch
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
                 pipeline_directory: str,
                 input: omegaconf.omegaconf,
                 model_config: omegaconf.omegaconf,
                 inference: omegaconf.omegaconf,
                 output: omegaconf.omegaconf) -> Dataset:
        Logger.debug(f'pipeline directory: {pipeline_directory}')
        dataset = None
        input_folder = 'input_folder' in input and input.input_folder or None
        image_paths = None
        if input_folder is not None:
            input_folder = get_path(pipeline_directory, input_folder)
            patterns = 'patterns' in input and input.patterns or ['**/*']
            recursive = 'recursive' in input and input.recursive or False
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
        batch_size = 'batch_size' in inference and inference.batch_size or 1
        filter_threshold = 'filter_threshold' in inference and inference.filter_threshold or 0.5
        output_texts = []
        checkpoint = model_config.checkpoint
        checkpoint = get_path(pipeline_directory, checkpoint)
        Logger.info(f'Loading checkpoint from {checkpoint}')
        model = WaifuDiffusionInterrogator(checkpoint)
        for i, images in enumerate(batch(train_images['image'], batch_size)):
            Logger.info(f'Processing batch {i}')
            reses = model.interrogate(images)
            for res in reses:
                tags = res[1]
                filter_tags = [k for k, v in tags.items() if v >= filter_threshold]
                if len(filter_tags) == 0:
                    continue
                Logger.info(f'Found {len(filter_tags)} tags')
                Logger.info(f'{",".join(filter_tags)}')
                output_texts.append(f'{",".join(filter_tags)}')
        Logger.info(f'Finished processing {len(output_texts)} samples')
        image_column = 'image_column' in output and output.image_column or 'image'
        caption_column = 'caption_column' in output and output.caption_column or 'caption'
        dataset = dataset.add_column(caption_column, output_texts)
        if(image_column != 'image'):
            dataset.rename_column('image', image_column)
        output_folder = output.output_folder
        output_folder = get_local_path(pipeline_directory, output_folder)
        Logger.info(f'Writing output to {output_folder}')
        os.makedirs(output_folder, exist_ok=True)
        dataset.save_to_disk(output_folder)

        output_as_image_folder = 'output_as_image_folder' in output and output.output_as_image_folder or False
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

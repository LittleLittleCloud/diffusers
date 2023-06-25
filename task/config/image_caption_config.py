from dataclasses import dataclass
from enum import Enum
from attr import attr, field
from typing import Union, List, Optional
from omegaconf import MISSING
from .dataset_config import *
from .model_config import *

@dataclass
class ImageCaptionInput:
    folder: Optional[ImageFolderConfig] = None
    dataset: Optional[DatasetConfig] = None

@dataclass
class ImageCaptionOutput:
    folder: Optional[ImageFolderConfig] = None
    dataset: Optional[DatasetConfig] = None

@dataclass
class BlipImageCaptionConfig:
    type: str = "blip_image_caption"
    input: ImageCaptionInput = MISSING
    output: ImageCaptionOutput = MISSING
    model: PretrainedModelConfig = MISSING
    # inference parameters
    device: str = "cuda"
    batch_size: int = 32
    trigger_word: Optional[str] = None
    image_column: Optional[str] = "image"
    caption_column: Optional[str] = "text"

@dataclass
class WDV1_4TaggerConfig(BlipImageCaptionConfig):
    type: str = "wdv1_4_tagger"
    filter_threshold: float = 0.5


@dataclass
class ImageCaptionTaggerConfig:
    type: str = "image_caption_tagger"
    input: ImageCaptionInput = MISSING
    output: ImageCaptionOutput = MISSING
    caption: BlipImageCaptionConfig = MISSING
    tagger: WDV1_4TaggerConfig = MISSING
    trigger_word: Optional[str] = None
    image_column: Optional[str] = "image"
    caption_column: Optional[str] = "text"
    device: str = "cuda"
    batch_size: int = 32

    
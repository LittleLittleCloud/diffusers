from dataclasses import dataclass, field
from enum import Enum
from typing import Union, List, Optional
from omegaconf import MISSING

class DatasetConfigType(Enum):
    ImageFolderConfig = "ImageFolderConfig"
    DatasetConfig = "DatasetConfig"

@dataclass
class ImageFolderConfig:
    type: DatasetConfigType = DatasetConfigType.ImageFolderConfig
    image_folder: str = MISSING
    patterns: Optional[List[str]] = field(default_factory=list)
    recursive: Optional[bool] = False
    image_column: Optional[str] = "image"
    image_path_column: Optional[str] = "image_path"
    label_column: Optional[str] = None
    metadata_name: Optional[str] = None

@dataclass
class DatasetConfig:
    type: DatasetConfigType = DatasetConfigType.DatasetConfig
    dataset_name: Optional[str] = None # huggingface repo id.
    dataset_config_name: Optional[str] = None # dataset folder
    image_colunm: Optional[str] = "image"
    label_colunm: Optional[str] = "text"
    push_to_hub: Optional[bool] = False # push to huggingface hub
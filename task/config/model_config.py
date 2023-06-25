from dataclasses import dataclass
from enum import Enum
from attr import attr, field
from typing import Union, List, Optional
from omegaconf import MISSING
from .dataset_config import *

@dataclass
class PretrainedModelConfig:
    model_name: str = MISSING # model name/path/folder

@dataclass
class HuggingFaceModelConfig:
    type: str = "huggingface_model"
    model_id: Optional[str] = None
    repo_id: str = MISSING
    push_to_hub: Optional[bool] = False

@dataclass
class LoraConfig:
    name: Union[PretrainedModelConfig, HuggingFaceModelConfig] = None
    is_safe_tensors: Optional[bool] = False

@dataclass
class LoadLoraConfig:
    type: str = "load_lora"
    lora: LoraConfig = MISSING
    weight: float = 0.5

@dataclass
class StableDiffusionModelConfig:
    type: str = "stable_diffusion"
    base_model: Union[str, HuggingFaceModelConfig] = None
    loras: List[LoadLoraConfig] = field(default_factory=list)
    sampler: str = 'ddim'
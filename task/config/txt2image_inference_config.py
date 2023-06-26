from .dataset_config import *
from .model_config import *
from dataclasses import dataclass

@dataclass
class Txt2ImageInput:
    folder: Optional[ImageFolderConfig] = None
    dataset: Optional[DatasetConfig] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    image: Optional[str] = None

@dataclass
class Txt2ImageOutput:
    folder: Optional[ImageFolderConfig] = None
    dataset: Optional[DatasetConfig] = None

@dataclass
class Txt2ImageInferenceConfig:
    output: ImageFolderConfig = MISSING
    model: StableDiffusionModelConfig = MISSING
    input: Txt2ImageInput = MISSING

    # inference parameters
    device: str = "cuda"
    dtype: str = "float16"
    num_images_per_prompt: int = 1
    cfg: float = 7.0
    step: int = 30
    seed: int = -1
    width: int = 512
    height: int = 768
    sampler: str = "ddim"
    prompt_column: Optional[str] = 'prompt'
    negative_prompt_column: Optional[str] = 'negative_prompt'
    output_image_column: Optional[str] = "output_image"

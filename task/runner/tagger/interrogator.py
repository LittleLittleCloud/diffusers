import os
import gc
import pandas as pd
import numpy as np
import torch
from typing import Tuple,List, Dict
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from huggingface_hub import hf_hub_download
from onnxruntime import InferenceSession
import torchvision.transforms.functional as F

class SquarePadWhite:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 255, 'constant')

class Interrogator:
    @staticmethod
    def postprocess_tags(
        tags: Dict[str, float],

        threshold=0.35,
        additional_tags: List[str] = [],
        exclude_tags: List[str] = [],
        sort_by_alphabetical_order=False,
        add_confident_as_weight=False,
        replace_underscore=False,
        replace_underscore_excludes: List[str] = [],
    ) -> Dict[str, float]:
        for t in additional_tags:
            tags[t] = 1.0

        # those lines are totally not "pythonic" but looks better to me
        tags = {
            t: c

            # sort by tag name or confident
            for t, c in sorted(
                tags.items(),
                key=lambda i: i[0 if sort_by_alphabetical_order else 1],
                reverse=not sort_by_alphabetical_order
            )

            # filter tags
            if (
                c >= threshold
                and t not in exclude_tags
            )
        }

        new_tags = []
        for tag in list(tags):
            new_tag = tag

            if replace_underscore and tag not in replace_underscore_excludes:
                new_tag = new_tag.replace('_', ' ')

            if add_confident_as_weight:
                new_tag = f'({new_tag}:{tags[tag]})'

            new_tags.append((new_tag, tags[tag]))
        tags = dict(new_tags)

        return tags

    def __init__(self, name: str) -> None:
        self.name = name

    def load(self):
        raise NotImplementedError()

    def unload(self) -> bool:
        unloaded = False

        if hasattr(self, 'model') and self.model is not None:
            del self.model
            unloaded = True
            print(f'Unloaded {self.name}')

        if hasattr(self, 'tags'):
            del self.tags

        return unloaded

    def interrogate(
        self,
        images: List[Image.Image],
    ) -> List[
        Tuple[
            Dict[str, float],  # rating confidents
            Dict[str, float]  # tag confidents
        ]
    ]:
        raise NotImplementedError()

class WaifuDiffusionInterrogator(Interrogator):
    def __init__(
        self,
        name: str,
        model_path='model.onnx',
        tags_path='selected_tags.csv',
        use_cpu=False,
        **kwargs
    ) -> None:
        super().__init__(name)
        self.model_path = model_path
        self.tags_path = tags_path
        self.kwargs = kwargs
        self.use_cpu = use_cpu

    def download(self) -> Tuple[os.PathLike, os.PathLike]:
        print(f"Loading {self.name} model file from")

        model_path = Path(hf_hub_download(
            repo_id=self.name, filename=self.model_path))
        tags_path = Path(hf_hub_download(
            repo_id=self.name, filename=self.tags_path))
        return model_path, tags_path

    def load(self) -> None:
        model_path, tags_path = self.download()

        # only one of these packages should be installed at a time in any one environment
        # https://onnxruntime.ai/docs/get-started/with-python.html#install-onnx-runtime
        # TODO: remove old package when the environment changes?

        # https://onnxruntime.ai/docs/execution-providers/
        # https://github.com/toriato/stable-diffusion-webui-wd14-tagger/commit/e4ec460122cf674bbf984df30cdb10b4370c1224#r92654958
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if self.use_cpu:
            providers.pop(0)

        self.model = InferenceSession(str(model_path), providers=providers)

        print(f'Loaded {self.name} model from {model_path}')

        self.tags = pd.read_csv(tags_path)

    def interrogate(
        self,
        images: List[Image.Image]
    ) -> List[
        Tuple[
            Dict[str, float],  # rating confidents
            Dict[str, float]  # tag confidents
        ]
    ]:
        # init model
        if not hasattr(self, 'model') or self.model is None:
            self.load()

        # code for converting the image and running the model is taken from the link below
        # thanks, SmilingWolf!
        # https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags/blob/main/app.py

        # convert an image to fit the model
        _, width, height, _ = self.model.get_inputs()[0].shape

        # alpha to white
        def transImage(image):
            image = image.convert('RGBA')
            new_image = Image.new('RGBA', image.size, 'WHITE')
            new_image.paste(image, mask=image)
            image = new_image.convert('RGB')
            return image
        trans = transforms.Compose([
            # transforms.Lambda(lambda x: transImage(x)),
            SquarePadWhite(),
            transforms.Resize((width, height)),
            transforms.ToTensor(),
            # [b, c, h, w] -> [b, h, w, c]
            transforms.Lambda(lambda x: x.permute(1, 2, 0)),
            # PIL RGB to OpenCV BGR
            transforms.Lambda(lambda x: x[:, :, [2, 1, 0]] * 255),
        ])
        images = [trans(image) for image in images]
        confidents = []
        for image in images:
            image = image.unsqueeze(0)
            image = image.numpy()
            # evaluate model
            input_name = self.model.get_inputs()[0].name
            label_name = self.model.get_outputs()[0].name
            confident = self.model.run([label_name], {input_name: image})[0]
            confidents.append(confident)

        res = []
        for confident in confidents:
            tags = self.tags[:][['name']]
            tags['confidents'] = confident[0]
            # first 4 items are for rating (general, sensitive, questionable, explicit)
            ratings = dict(tags[:4].values)
            # rest are regular tags
            tags = dict(tags[4:].values)

            res.append((ratings, tags))

        return res
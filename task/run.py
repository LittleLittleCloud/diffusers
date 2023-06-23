import argparse
import omegaconf
from .runner.txt2img_inference import Txt2ImgInferenceRunner
from .runner.txt2img_train_lora import Txt2ImgLoraTraningRunner
from .runner.blip_image_caption import BlipImageCaptionRunner
from .runner.image_caption_tagger import ImageCaptionTaggerRunner
from .log import get_logger
import logging
import os
Logging = get_logger(__name__)
Logging.setLevel(logging.DEBUG)
AVAILABLE_RUNNERS = {
    Txt2ImgInferenceRunner.name: Txt2ImgInferenceRunner,
    Txt2ImgLoraTraningRunner.name: Txt2ImgLoraTraningRunner,
    BlipImageCaptionRunner.name: BlipImageCaptionRunner,
    ImageCaptionTaggerRunner.name: ImageCaptionTaggerRunner,
}
def parse_args():
    parser = argparse.ArgumentParser(description='stable diffusion pipeline runner')
    parser.add_argument(
        '--pipeline',
        type=str,
        default=None,
        required=True,
        help='The pipeline to use'
    )

    return parser.parse_args()

def main(args):
    pipeline = args.pipeline
    Logging.info(f'Loading pipeline from {pipeline}')
    pipeline_path = os.path.join(os.getcwd(), pipeline) if not os.path.isabs(pipeline) else pipeline
    Logging.debug(f'pipeline path: {pipeline_path}')
    cfg = omegaconf.OmegaConf.load(pipeline)
    for task in cfg:
        task['_pipeline_path'] = pipeline_path
        Logging.info(f'Running task: {task.name}')
        runner = AVAILABLE_RUNNERS[task.task]()
        runner.execute(task)

if __name__ == '__main__':
    args = parse_args()
    main(args)
    



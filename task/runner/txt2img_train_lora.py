import math
import random
import numpy as np
import omegaconf
import torch
import tqdm
from diffusers import StableDiffusionPipeline
from diffusers.utils.loras import load_lora_weights
from task.log import get_logger
from task.runner.utils import load_pipeline
import os
from task.runner.runner import Runner
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from task.runner.utils import get_path, get_local_path
from diffusers.optimization import get_scheduler
import torch.nn.functional as F
import datasets
import transformers
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import load_dataset, load_from_disk
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from torchvision import transforms

Logger = get_logger(__name__)

def main(cwd: str, config: omegaconf.omegaconf):
    pipeline_path = config._pipeline_path
    output_dir = get_local_path(cwd, config.output_folder)
    Logger.info(f'Output directory: {output_dir}')
    logging_dir = os.path.join(output_dir, config.logging_folder)
    Logger.info(f'Logging to {logging_dir}')
    checkpoint_total_limit = 'checkpoints_total_limit' in config and config.checkpoints_total_limit or None
    gradient_accumulation_steps = 'gradient_accumulation_steps' in config and config.gradient_accumulation_steps or 1
    Logger.info(f'Checkpoint total limit: {checkpoint_total_limit}')
    Logger.info(f'Gradient accumulation steps: {gradient_accumulation_steps}')
    
    accelerator_project_config = ProjectConfiguration(
        logging_dir=logging_dir,
        total_limit=checkpoint_total_limit)
    accelerator = Accelerator(
        log_with='tensorboard',
        gradient_accumulation_steps=gradient_accumulation_steps,
        project_config=accelerator_project_config,
    )

    Logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    seed = 'seed' in config and config.seed or None
    if seed is not None:
        Logger.info(f'Setting seed to {seed}')
        set_seed(seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    model = load_pipeline(
        cfg=config.model,
        directory_path=cwd,
        logger=Logger,)
    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    # )
    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    # )
    # vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    # unet = UNet2DConditionModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    # )
    noise_scheduler = model.scheduler
    tokenizer = model.tokenizer
    text_encoder = model.text_encoder
    vae = model.vae
    unet = model.unet

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)

    text_encoder.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # now we will add new LoRA weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Set correct lora layers
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

    unet.set_attn_processor(lora_attn_procs)

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    lora_layers = AttnProcsLayers(unet.attn_processors)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    allow_tf32 = 'allow_tf32' in config and config.allow_tf32 or False
    if allow_tf32:
        Logger.info("Enabling TF32 for faster training")
        torch.backends.cuda.matmul.allow_tf32 = True

    scale_lr = 'scale_lr' in config and config.scale_lr or False
    learning_rate = config.learning_rate
    train_batch_size = config.batch_size
    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    Logger.info(f"Learning rate: {learning_rate}")
    Logger.info(f"batch size: {train_batch_size}")

    use_8bit_adam = 'use_8bit_adam' in config and config.use_8bit_adam or False
    Logger.info(f"Using 8-bit Adam: {use_8bit_adam}")
    # Initialize the optimizer
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    adam_beta1 = 'adam_beta1' in config and config.adam_beta1 or 0.9
    adam_beta2 = 'adam_beta2' in config and config.adam_beta2 or 0.999
    adam_epsilon = 'adam_epsilon' in config and config.adam_epsilon or 1e-8
    adam_weight_decay = 'adam_weight_decay' in config and config.adam_weight_decay or 1e-2
    
    Logger.info(f"Adam beta1: {adam_beta1}")
    Logger.info(f"Adam beta2: {adam_beta2}")
    Logger.info(f"Adam epsilon: {adam_epsilon}")
    Logger.info(f"Adam weight decay: {adam_weight_decay}")
    optimizer = optimizer_cls(
        lora_layers.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.

    dataset_name = 'dataset_name' in config and config.dataset_name or None
    dataset_name = get_path(cwd, dataset_name)
    dataset_config_name = 'dataset_config_name' in config and config.dataset_config_name or None
    cache_dir = 'cache_dir' in config and config.cache_dir or None
    if dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        Logger.info(f"Loading dataset {dataset_name} with config {dataset_config_name}")
        
        try:
            dataset = load_dataset(
            dataset_name,
            dataset_config_name,
            cache_dir=cache_dir,
            )
        except ValueError:
            Logger.info(f"Loading dataset {dataset_name} with config {dataset_config_name} failed. Trying to load from disk.")
            dataset = load_from_disk(dataset_name)
    else:
        train_data_dir = 'train_data_dir' in config and config.train_data_dir or None
        train_data_dir = get_path(cwd, train_data_dir)
        Logger.info(f"Loading dataset from {train_data_dir}")
        data_files = {}
        data_files = os.path.join(train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    # 6. Get the column names for input/target.
    image_column = 'image_column' in config and config.image_column or None
    caption_column = 'caption_column' in config and config.caption_column or None
    if image_column is None:
        raise ValueError('image_column must be provided in the config file')
    if caption_column is None:
        raise ValueError('caption_column must be provided in the config file')
    Logger.info(f"Image column: {image_column}")
    Logger.info(f"Caption column: {caption_column}")
    
    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.

    resolution = 'resolution' in config and config.resolution or [512, 512]
    center_crop = 'center_crop' in config and config.center_crop or False
    random_flip = 'random_flip' in config and config.random_flip or False
    Logger.info(f"Resolution: {resolution}")
    Logger.info(f"Center crop: {center_crop}")
    Logger.info(f"Random flip: {random_flip}")

    train_transforms = transforms.Compose(
        [
            transforms.Resize([max(resolution)], interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        max_train_samples = 'max_train_samples' in config and config.max_train_samples or None
        if max_train_samples is not None and max_train_samples < len(dataset):
            dataset = dataset.shuffle(seed=seed).select(range(max_train_samples))
        # Set the training transforms
        train_dataset = dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    dataloarder_num_workers = 'dataloader_num_workers' in config and config.dataloader_num_workers or 4
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        num_workers=dataloarder_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_train_epochs = 'num_train_epochs' in config and config.num_train_epochs or 1
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = 'max_train_steps' in config and config.max_train_steps or None
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    Logger.info(f"max_train_steps: {max_train_steps}")
    Logger.info(f"num_train_epochs: {num_train_epochs}")

    lr_scheduler = 'lr_scheduler' in config and config.lr_scheduler or 'constant'
    lr_warmup_steps = 'lr_warmup_steps' in config and config.lr_warmup_steps or 500
    Logger.info(f"lr_scheduler: {lr_scheduler}")
    Logger.info(f"lr_warmpup_steps: {lr_warmup_steps}")
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_layers, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune")

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    Logger.info("***** Running training *****")
    Logger.info(f"  Num examples = {len(train_dataset)}")
    Logger.info(f"  Num Epochs = {num_train_epochs}")
    Logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    Logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    Logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    Logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    resume_from_checkpoint = 'resume_from_checkpoint' in config and config.resume_from_checkpoint or None
    Logger.info(f"  Resume from checkpoint {resume_from_checkpoint}")
    
    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm.tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                noise_offset = 'noise_offset' in config and config.noise_offset or 0.0
                # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                noise += noise_offset * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get the target for loss depending on the prediction type
                prediction_type = 'prediction_type' in config and config.prediction_type or None
                if prediction_type is not None:
                    Logger.info(f"prediction_type: {prediction_type}")
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                snr_gamma = 'snr_gamma' in config and config.snr_gamma or None
                if snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(timesteps)
                    mse_loss_weights = (
                        torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                max_grad_norm = 'max_grad_norm' in config and config.max_grad_norm or 1.0
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                
                checkpoint_steps = 'checkpoint_steps' in config and config.checkpoint_steps or 1000
                if global_step % checkpoint_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        Logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        if accelerator.is_main_process:
            validation_prompt = 'validation_prompt' in config and config.validation_prompt or None
            validation_negative_prompt = 'validation_negative_prompt' in config and config.validation_negative_prompt or None
            validation_epochs = 'validation_epochs' in config and config.validation_epochs or 1
            num_validation_images = 'num_validation_images' in config and config.num_validation_images or 4
            if validation_prompt is not None and (epoch % validation_epochs == 0 or epoch == num_train_epochs - 1):
                Logger.debug(
                    f"Running validation... \n Generating {num_validation_images} images with prompt:"
                    f" {validation_prompt}."
                )
                # create pipeline
                unet = accelerator.unwrap_model(unet)
                # pipeline = DiffusionPipeline.from_pretrained(
                #     args.pretrained_model_name_or_path,
                #     unet=accelerator.unwrap_model(unet),
                #     revision=args.revision,
                #     torch_dtype=weight_dtype,
                # )
                pipeline = model
                pipeline.unet = unet
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                generator = torch.Generator(device=accelerator.device)
                if seed is not None:
                    generator = generator.manual_seed(seed)
                images = []
                for _ in range(num_validation_images):
                    images.append(
                        pipeline(
                        prompt=validation_prompt,
                        negative_prompt=validation_negative_prompt,
                        num_inference_steps=30,
                        generator=generator).images[0]
                    )

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                
                torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_name = 'output_name' in config and config.output_name or None
        save_as_safetensors = 'save_as_safe_tensors' in config and config.save_as_safe_tensors or False
        Logger.info(f"Saving lora layers to {output_dir}")
        Logger.info(f"Saving lora layers as safe tensors: {save_as_safetensors}")
        unet = unet.to(torch.float32)
        unet.save_attn_procs(
            save_directory=output_dir,
            safe_serialization=save_as_safetensors,
            weight_name=output_name)

    accelerator.end_training()

class Txt2ImgLoraTraningRunner(Runner):
    name: str = 'txt2img_lora_training.v0'
    def execute(self, cwd:str, config: omegaconf.omegaconf):
        main(cwd, config)
        
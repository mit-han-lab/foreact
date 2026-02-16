import datasets
import os
import shutil
import torch
import transformers
import yaml
import PIL.Image

from accelerate.utils import release_memory
from dataclasses import dataclass, field
from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint
from PIL import PngImagePlugin

from dataloaders.dataset_finetune import get_train_datasets
from models.visualforesight import VisualForesightConfig, VisualForesight
from utils.trainer_utils import find_newest_checkpoint, possible_override_args, ModelCallback

datasets.disable_caching()
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_PROJECT"] = "VisualForesight"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

PIL.Image.MAX_IMAGE_PIXELS = None
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)


@dataclass
class OverrideArguments:
    config_file: str = None


@dataclass
class ModelArguments:
    mllm_id: str = "google/gemma-2-2b-it"
    diffusion_model_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    vae_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    noise_scheduler_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    scheduler_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    max_input_text_tokens: int = 256
    vae_downsample_f: int = 32
    in_channels: int = 32
    system_prompt: str = "You are a robot and should focus on your actions. Generate a new image that meets the user's instruction while maintaining consistency with the original input where appropriate."
    _gradient_checkpointing: bool = True
    modules_to_freeze: tuple[str] = ()
    modules_to_unfreeze: tuple[str] = ()


@dataclass
class DataArguments:
    data_path: str = "data/realworld"
    camera_key: str = "observation.images.head_left_rgb"
    target_image_size: tuple[int, int] = (480, 640)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = "output"
    per_device_train_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    optim: str = "adamw_torch"
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 0.5
    lr_scheduler_type: str = "cosine_with_min_lr"
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {"min_lr": 1e-5})
    warmup_steps: int = 5000
    logging_steps: int = 1
    save_steps: int = 1000
    save_total_limit: int = 1000
    restore_callback_states_from_checkpoint: bool = True
    seed: int = 42
    bf16: bool = True
    tf32: bool = True
    dataloader_num_workers: int = 4
    datasets_num_proc: int = os.getenv("OMP_NUM_THREADS", 12)
    dataloader_persistent_workers: bool = False
    dataloader_pin_memory: bool = True
    dataloader_drop_last: bool = True
    remove_unused_columns: bool = False
    run_name: str = "test"
    report_to: str = "wandb"
    ddp_find_unused_parameters: bool = False
    overwrite_output_dir: bool = False
    resume_from_checkpoint: str = None

    def __post_init__(self):
        try:
            self = possible_override_args(override_args, self)
        except (FileNotFoundError, yaml.YAMLError) as exc:
            print(f"Failed to load override config: {exc}")
        super().__post_init__()


if __name__ == "__main__":
    override_parser = transformers.HfArgumentParser((OverrideArguments))
    override_args = override_parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )[0]
    parser = transformers.HfArgumentParser(
        (OverrideArguments, ModelArguments, DataArguments, TrainingArguments)
    )
    _, model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args, data_args = possible_override_args(override_args, model_args, data_args)

    assert (
        data_args.target_image_size[0] % model_args.vae_downsample_f == 0 and data_args.target_image_size[1] % model_args.vae_downsample_f == 0
    ), f"Image size must be divisible by {model_args.vae_downsample_f}"
    input_size = (data_args.target_image_size[0] // model_args.vae_downsample_f, data_args.target_image_size[1] // model_args.vae_downsample_f)

    if training_args.resume_from_checkpoint is not None:
        training_args.resume_from_checkpoint = find_newest_checkpoint(
            training_args.resume_from_checkpoint
        )
        model = VisualForesight.from_pretrained(
            training_args.resume_from_checkpoint,
            input_size=input_size,
            ignore_mismatched_sizes=True,
            **model_args.__dict__,
        )
    else:
        model = VisualForesight(
            config=VisualForesightConfig(
                input_size=input_size,
                **model_args.__dict__,
            ),
        )

    with training_args.main_process_first(local=False):
        train_dataset, collate_fn = get_train_datasets(
            data_args,
            model.get_tokenize_fn(),
            model.get_tokenizer(),
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=[ModelCallback()],
    )

    training_args.output_dir = str(
        os.path.join(training_args.output_dir, training_args.run_name)
    )
    if trainer.is_world_process_zero():
        if training_args.overwrite_output_dir and os.path.exists(
            training_args.output_dir
        ):
            shutil.rmtree(training_args.output_dir)
        print(f"Training dataset size: {len(train_dataset)}")

    while (
        trainer.state.epoch is None
        or (training_args.num_train_epochs - trainer.state.epoch) > 0.01
    ):
        if trainer.state.epoch is not None:
            trainer.control.should_training_stop = False
            trainer.args.eval_on_start = False
            trainer.model = model
            (trainer.model_wrapped,) = release_memory(trainer.model_wrapped)
            trainer.model_wrapped = trainer.model
        last_checkpoint = None
        if (
            os.path.isdir(training_args.output_dir)
            and not training_args.overwrite_output_dir
        ):
            last_checkpoint = get_last_checkpoint(training_args.output_dir)

        trainer.train(resume_from_checkpoint=last_checkpoint)

import os
import yaml

from tabulate import tabulate
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint


def possible_override_args(override_args, *args):
    if hasattr(override_args, "config_file") and override_args.config_file is not None:
        yaml_file = os.path.join("configs", override_args.config_file)
        with open(yaml_file, "r") as file:
            config = yaml.safe_load(file)

        for arg in args:
            for key, value in config.items():
                if hasattr(arg, key):
                    setattr(arg, key, value)
    return args[0] if len(args) == 1 else args


def find_newest_checkpoint(checkpoint_path):
    if os.path.isdir(checkpoint_path) and any(
        x.endswith(("pt", "safetensors", "pth")) for x in os.listdir(checkpoint_path)
    ):
        return checkpoint_path

    else:
        return get_last_checkpoint(checkpoint_path)


class ModelCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model, **kwargs):
        if state.is_world_process_zero:
            stat = []
            for i, (n, p) in enumerate(model.named_parameters()):
                stat.append([i, n, p.shape, p.dtype, p.requires_grad])
            print(
                tabulate(stat, headers=["idx", "name", "shape", "dtype", "trainable"])
            )

            num_params = sum(p.numel() for p in model.transformer.parameters()) / 1e6
            print(f"transformer num of params: {num_params:.2f}M")
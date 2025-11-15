import os
from pathlib import Path, PosixPath

import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

from framevision import autoloading as al

torch.serialization.add_safe_globals([PosixPath])

# Register a custom resolver for 'eval'
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)

# Define paths
ROOT_DIR = Path(__file__).absolute().parent.parent.parent
SCRIPT_DIR = ROOT_DIR / "scripts" / "utils"
CONFIG_DIR = ROOT_DIR / "configs"


def load_experiment_config(name: str = None, dataset_path: str = None, instantiate_objects: bool = False, overrides: list[str] = []):
    relative_config_path = os.path.relpath(CONFIG_DIR, SCRIPT_DIR)
    initialize(version_base="1.3", config_path=relative_config_path)

    overrides = [f"experiments={name}"] + overrides if name else overrides
    config = compose(config_name="default", overrides=overrides)

    config = OmegaConf.to_container(config)
    # Adding the dataset path to the configuration
    if dataset_path is not None:
        config["dataset-path"] = dataset_path

    # Setting the logs folder to have an intellegible name
    config["logs-path"] = ROOT_DIR / f"{config['logs-path']}/{config['experiment-name']}"

    if config["logs-path"].exists():
        config["logs-path"] = add_versioning_to_path(config["logs-path"])

    config["logs-path"].mkdir(parents=True)
    config["checkpoints-folder"] = config["logs-path"] / "checkpoints"

    if instantiate_objects:
        return resolve_and_instantiate(config)

    return config


def resolve_and_instantiate(config: dict):
    # Resolve the variables in the configuration
    omega_config = OmegaConf.create(config)
    OmegaConf.resolve(omega_config)
    config = OmegaConf.to_container(omega_config)

    # Instantiate objects from the configuration
    instantiated_objects = instantiate(config, _convert_="all")

    return config, instantiated_objects


def resume_experiment_config(resume_from: str | Path, dataset_path: str = None, instantiate_objects: bool = True):
    if not instantiate_objects:
        raise ValueError("This function is only meant to be used with `instantiate_objects=True`")

    checkpoint_path = al.find_ckpt_path(resume_from)

    if checkpoint_path is None:
        msg = f"Checkpoint '{resume_from}' is not a path, and it was not found in the global checkpoint folder {al.global_ckpt_folder}."
        raise ValueError(msg)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Initialize the model given the hydra configuration
    config = checkpoint["hyper_parameters"]["hydra"]

    if "dataset-path" in config and dataset_path is not None:
        config["dataset-path"] = dataset_path
        config["dataset"]["root_dir"] = dataset_path

    config["ckpt-path"] = checkpoint_path

    # Instantiate objects from the configuration
    instantiated_objects = instantiate(config)

    # Load the model state
    instantiated_objects.model.load_state_dict(checkpoint["state_dict"], strict=False)

    return config, instantiated_objects


def add_versioning_to_path(path: Path):
    attempt = path
    i = 1
    while attempt.exists():
        attempt = path.parent / f"{path.stem}_v{i}{path.suffix}"
        i += 1
    return attempt

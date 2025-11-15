from pathlib import Path, PosixPath

import torch
from hydra.utils import instantiate

torch.serialization.add_safe_globals([PosixPath])
root_folder = Path(__file__).parent.parent.parent.parent.resolve()
global_ckpt_folder = root_folder / "checkpoints"


def load_model(name: str | Path, attribute: str = None, **kwargs):
    """
    Load a model from a specific checkpoint.
    If `name` is a path, it will load the model from that path.
    If the `name` is an experiment of the form `project_name/run_name`, it will look for it in the global checkpoint folder.

    Args:
        run_name (str): The name of the run or path to the checkpoint file.
        attribute (str): The attribute to retrieve from the model. If None, returns the entire model.
        **kwargs: Additional arguments to pass to the model instantiation.

    Returns:
        The loaded model or the specified attribute of the model.
    """
    # Load from path if it is a path
    path = Path(name) if isinstance(name, str) else name
    if path.exists():
        return load_model_from_file(path, attribute, **kwargs)

    # Otherwise, load from the global checkpoint folder
    global_path = find_ckpt_path(name)
    if global_path:
        return load_model_from_file(global_path, attribute, **kwargs)

    error_msg = f"Checkpoint {name} is not a path, and it was not found in the global checkpoint folder {global_ckpt_folder}."
    raise ValueError(error_msg)


def load_model_from_file(checkpoint_path: Path, attribute: str = None, **kwargs):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Initialize the model given the hydra configuration
    config = checkpoint["hyper_parameters"]["hydra"]["model"]

    # If attribute is given, set all the other attributes to None
    if attribute is not None:
        necessary_fields = attribute.split(".") + ["_target_"]
        config = {k: 0 if k not in necessary_fields else v for k, v in config.items()}

    model = instantiate(config)

    # If the model was compiled, we remove the state_dict ._orig_mod prefix
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if "._orig_mod" in k:
            k = k.replace("._orig_mod", "")
        new_state_dict[k] = v

    # Load the model state
    model.load_state_dict(new_state_dict, strict=True)

    if attribute is not None:
        attribute_chain = attribute.split(".")
        for attr in attribute_chain:
            model = getattr(model, attr)

        return model

    # Save hydra hyperparams once again
    model.store_hydra_configuration(config)

    return model


def find_ckpt_path(run_name: str) -> Path | None:
    folder = global_ckpt_folder / run_name
    if not folder.exists():
        return None

    checkpoint_files = list(folder.glob("*.ckpt"))
    if len(checkpoint_files) > 1:
        raise ValueError(f"Found more than one checkpoint file in {folder}, please specify the file name or remove the extra files.")
    if len(checkpoint_files) == 0:
        raise ValueError(f"No checkpoint files found in {folder}")

    return checkpoint_files[0]

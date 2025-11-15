import shutil
from datetime import datetime
from pathlib import Path

import lightning as L
import rich
import typer
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichModelSummary
from utils import load_experiment_config

from framevision.autoloading import global_ckpt_folder

CONSOLE = rich.get_console()

app = typer.Typer()

#torch.set_float32_matmul_precision('high')

@app.command()
def main(
    data: Path = typer.Option(..., help="Path to the dataset"),
    experiment: str = typer.Option(..., help="Experiment name in configs/experiments"),
    override: list[str] = typer.Option([], help="Overrides for the configuration"),
    logger: str = typer.Option("wandb", help="Logger to use (`wandb` or `tensorboard` or `none`)"),
):
    L.seed_everything(42)

    hparams, objects = load_experiment_config(experiment, data, instantiate_objects=True, overrides=override)
    log_info(hparams, data)

    logger = setup_logger(hparams["logs-path"], hparams["experiment-name"], hparams, hparams["project-name"], logger=logger)
    callbacks = setup_callbacks(hparams["checkpoints-folder"], hparams["logs-path"], hparams["metric-to-track"])

    model = objects["model"]
    datamodule = objects["dataset"]
    train_params = objects["training"]

    # Save hydra configuration in the model checkpoint
    model.store_hydra_configuration(hparams)

    trainer = L.Trainer(logger=logger, callbacks=callbacks, **train_params["trainer"])
    trainer.fit(model, datamodule, ckpt_path=hparams.get("ckpt-path"))

    # Save the final checkpoint in the global checkpoint folder
    save_ckpt_to_global_folder(trainer.checkpoint_callback.best_model_path, hparams["project-name"], hparams["experiment-name"])


def setup_callbacks(ckpt_dir: Path, log_dir: Path, metric_to_track: str) -> list[L.Callback]:
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        RichModelSummary(max_depth=2),
        ModelCheckpoint(monitor=metric_to_track, save_top_k=1, mode="min", dirpath=ckpt_dir),
    ]

    return callbacks


def setup_logger(log_dir: Path, exp_name: str, hparams: dict, project_name: str = "", logger: str = "wandb"):
    if not logger or logger == "none":
        return None

    project_name = project_name or "frame"
    if logger == "tensorboard":
        from lightning.pytorch.loggers import TensorBoardLogger

        return TensorBoardLogger(save_dir=log_dir, name=exp_name, version=0)
    elif logger == "wandb":
        from lightning.pytorch.loggers import WandbLogger

        return WandbLogger(
            name=exp_name,
            project=project_name,
            config=hparams,
            save_dir=log_dir,
            log_model="all",
            checkpoint_name="checkpoint",
        )
    else:
        raise ValueError(f"Logger {logger} not supported. Use 'wandb' or 'tensorboard'.")


def log_info(hparams: dict, data: Path):
    CONSOLE.print(f"Starting experiment: {hparams['experiment-name']}", style="bold green")
    CONSOLE.print(f"Using dataset: {data.resolve()}", style="bold green")
    CONSOLE.print(f"Logging to: {hparams['logs-path']}", style="bold green")
    CONSOLE.print(f"Checkpoints to: {hparams['checkpoints-folder']}", style="bold green")


def save_ckpt_to_global_folder(local_ckpt_path: Path, project_name: str, experiment_name: str):
    final_ckpt_folder = global_ckpt_folder / experiment_name
    final_ckpt_folder.mkdir(parents=True, exist_ok=True)
    final_ckpt_path = final_ckpt_folder / "checkpoint.ckpt"

    if final_ckpt_path.exists():
        # If the checkpoint already exists, rename it with a timestamp
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        new_ckpt_name = f"checkpoint_{now}.ckpt"
        final_ckpt_path = final_ckpt_folder / new_ckpt_name

    # Copy the checkpoint to the global folder
    shutil.copy(local_ckpt_path, final_ckpt_path)
    CONSOLE.print(f"Final Checkpoint copied to: {final_ckpt_path.resolve()}", style="bold green")


if __name__ == "__main__":
    app()

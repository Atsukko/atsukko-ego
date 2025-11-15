from itertools import chain
from pathlib import Path

import lightning as L
import numpy as np
import torch
import typer
from torch import Tensor
from tqdm import tqdm
from utils import resume_experiment_config

app = typer.Typer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


@app.command()
def main(
    data: Path = typer.Option(..., help="Path to the dataset"),
    load: str = typer.Option(..., help="Resume from a specific checkpoint"),
    cache_name: str = typer.Option(..., help="Name of the cache file"),
    out_key: str = typer.Option("joints_3D", help="Output key for the 3D joints"),
    only_val: bool = typer.Option(False, help="Only cache the validation set"),
):
    L.seed_everything(42)

    hparams, objects = resume_experiment_config(load, data, instantiate_objects=True)

    network = objects.model.network.to(device).eval()
    datamodule = objects.dataset
    datamodule.setup()

    train_dataloader = datamodule.train_dataloader(drop_last=False)
    val_dataloader = datamodule.val_dataloader()
    all_iterator = val_dataloader if only_val else chain(train_dataloader, val_dataloader)
    iterator_len = len(val_dataloader) if only_val else len(train_dataloader) + len(val_dataloader)

    results = dict(actions=[], sequences=[], idxes=[], motions=[], ground_truth=[])

    for batch_idx, batch in tqdm(enumerate(all_iterator), total=iterator_len, desc="Processing batches"):
        kwargs = unpack_batch_data(batch, device) if "images" in batch else unpack_cached_batch_data(batch, device)

        with torch.no_grad():
            output = network(**kwargs)

        assert out_key in output, f"Output key {out_key} not found in the network output"

        idx, sequence, action = batch["meta"]["idx"], batch["meta"]["sequence"], batch["meta"]["action"]
        assert idx.shape[-1] == 1, "The caching algorithm expects the time dimension to be 1"

        results["actions"].append(action)
        results["sequences"].append(sequence)
        results["idxes"].append(idx.detach().cpu())
        results["motions"].append(output[out_key].detach().cpu())

    print("Remapping the results...")
    all_motions = remap_as_dict(results["motions"], results["sequences"], results["actions"], results["idxes"])

    for sequence in tqdm(all_motions, desc="Saving"):
        for action in all_motions[sequence]:
            cache_folder = data / sequence / "actions" / action / "cache" / cache_name
            cache_folder.mkdir(parents=True, exist_ok=True)
            cache_file = cache_folder / f"{out_key}.npz"
            np.savez(cache_file, motions=all_motions[sequence][action].numpy())


def remap_as_dict(
    motions: list[Tensor],
    sequences: list[str],
    actions: list[str],
    idxes: list[Tensor],
):
    sequences = [item for sublist in sequences for item in sublist]
    actions = [item for sublist in actions for item in sublist]
    idxes = torch.cat(idxes, dim=0)
    motions = torch.cat(motions, dim=0)

    all_motions = {}

    for seq, act, idxs, jts3D in tqdm(zip(sequences, actions, idxes, motions), total=len(sequences), desc="Remapping"):
        all_motions.setdefault(seq, {}).setdefault(act, {})
        for idx, jt3D in zip(idxs, jts3D):
            all_motions[seq][act][idx.item()] = jt3D

    for seq in tqdm(all_motions, desc="Flattening"):
        for act in all_motions[seq]:
            idx_to_pose = all_motions[seq][act]
            all_motions[seq][act] = torch.stack([idx_to_pose[idx] for idx in sorted(idx_to_pose)])

    return all_motions


def unpack_batch_data(batch, device):
    return dict(
        images=batch["images"].to(device, non_blocking=True),
        K=batch["intrinsics_norm"]["K"].to(device, non_blocking=True),
        d=batch["intrinsics_norm"]["d"].to(device, non_blocking=True),
        left2middle=batch["transforms"]["egocam_left_to_egocam_middle"].to(device, non_blocking=True),
        right2middle=batch["transforms"]["egocam_right_to_egocam_middle"].to(device, non_blocking=True),
        middle2world=batch["poses"]["vr"]["egocam_middle"].to(device, non_blocking=True),
    )


def unpack_cached_batch_data(batch, device):
    return dict(
        joints_3D=batch["cache"]["joints_3D_cc"].to(device, non_blocking=True),
        left2middle=batch["transforms"]["egocam_left_to_egocam_middle"].to(device, non_blocking=True),
        right2middle=batch["transforms"]["egocam_right_to_egocam_middle"].to(device, non_blocking=True),
        middle2world=batch["poses"]["vr"]["egocam_middle"].to(device, non_blocking=True),
    )


if __name__ == "__main__":
    app()

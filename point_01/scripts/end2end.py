from pathlib import Path

import lightning as L
import torch
import torchvision
import typer
from tqdm import tqdm

import framevision
from framevision.model import Frame
from framevision.pl_wrappers import FrameDataModule

app = typer.Typer()


@app.command()
def main(
    data: Path = typer.Option(..., help="Path to the data directory"),
    backbone_path: str = typer.Option("backbone", help="Backbone to use for evaluation. Name of the exp or the path to the checkpoint."),
    stf_path: str = typer.Option("stf", help="STF to use for evaluation. Can be the name of the experiment or the path to the checkpoint."),
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run this script on a machine with an NVIDIA GPU.")

    device = torch.device("cuda")
    L.seed_everything(42)

    backbone = framevision.autoloading.load_model(backbone_path, attribute="network").eval().cuda()
    stf = framevision.autoloading.load_model(stf_path, attribute="network").eval().cuda()

    backbone = torch.compile(backbone, mode="max-autotune", fullgraph=True)
    stf = torch.compile(stf, mode="max-autotune", fullgraph=True)

    model = Frame(backbone, stf).eval().cuda()

    datamodule = prepare_datamodule(data)

    preds, gt = [], []
    timings, current_kwargs, current_gts = [], [], []
    undersampling_factor = stf.undersampling_factor if hasattr(stf, "undersampling_factor") else stf._orig_mod.undersampling_factor

    for i, batch in tqdm(enumerate(datamodule.val_dataloader()), desc="Processing batches", total=len(datamodule.val_dataloader())):
        sequence, action = batch["meta"]["sequence"][0], batch["meta"]["action"][0]
        if i == 0:
            current_seq_act = (sequence, action)

        # Accumulate all the batches of the same sequence and action
        if (sequence, action) == current_seq_act:
            # We only process every undersampling_factor-th frame, to match how the model was trained.
            if i % undersampling_factor != 0:
                continue

            kwargs = unpack_batch_data(batch, device)
            joints3Dgt = batch["body_tracking"]["joints_3D"].to(device)

            current_kwargs.append(kwargs)
            current_gts.append(joints3Dgt)
            continue

        current_seq_act = (sequence, action)
        model.reset()

        for kwargs, joints3Dgt in zip(current_kwargs, current_gts):
            with torch.no_grad():
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

                joints3Dwr = model(**kwargs)

                end_event.record()
                torch.cuda.synchronize()
                duration = start_event.elapsed_time(end_event)

            if model.is_warming_up():
                continue

            joints3Dwr = joints3Dwr.squeeze().cpu().half()
            joints3Dgt = joints3Dgt.squeeze().cpu().half()
            preds.append(joints3Dwr)
            gt.append(joints3Dgt)
            timings.append(duration)

        # Reset the current batch
        current_kwargs = [unpack_batch_data(batch, device)]
        current_gts = [batch["body_tracking"]["joints_3D"].to(device)]

    pred_tensor, gt_tensor = torch.cat(preds), torch.cat(gt)
    errors = pred_tensor - gt_tensor

    # Since we skip every undersampling_factor-th frame, the MPJPE computed here might slightly differ from the one computed in eval.py.
    mpjpe = torch.norm(errors, dim=-1).mean() * 1000
    print(f"MPJPE: {mpjpe:.2f}mm")

    # Remove top 10% and bottom 10% of durations
    durations = torch.tensor(sorted(timings))
    n = int(len(durations) * 0.1)
    durations = durations[n:-n]

    mean_duration = durations.mean()
    std_duration = durations.std()
    print(f"Mean duration: {mean_duration:.2f}ms ± {std_duration:.2f}ms")


def unpack_batch_data(batch, device):
    images = batch["images"].to(device).squeeze()
    K = batch["intrinsics_norm"]["K"].to(device).squeeze()
    d = batch["intrinsics_norm"]["d"].to(device).squeeze()
    lTm = batch["transforms"]["egocam_left_to_egocam_middle"].to(device).squeeze()
    rTm = batch["transforms"]["egocam_right_to_egocam_middle"].to(device).squeeze()
    mTw = batch["poses"]["vr"]["egocam_middle"].to(device).squeeze()
    return dict(images=images, K=K, d=d, left2middle=lTm, right2middle=rTm, middle2world=mTw)


def prepare_datamodule(data: Path):
    processing = torchvision.transforms.Compose(
        [
            framevision.processing.Resize((256, 256)),
            framevision.processing.NormalizeImages(),
            framevision.processing.NormalizeJoints2D(),
            framevision.processing.NormalizeIntrinsics(),
        ]
    )

    datamodule = FrameDataModule(
        root_dir=data,
        batch_size=1,
        num_workers=0,
        split=dict(train="others", val=["test_actor00_seq1", "test_actor00_seq2", "test_actor01_seq1", "test_actor01_seq2"]),
        split_by="sequences",
        train_processing=processing,
        test_processing=processing,
    )
    datamodule.setup()
    return datamodule


if __name__ == "__main__":
    app()

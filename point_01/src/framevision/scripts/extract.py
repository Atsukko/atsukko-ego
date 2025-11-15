import concurrent.futures
import os
import zipfile
from pathlib import Path

import cv2
import typer
from rich.console import Console
from tqdm.rich import tqdm

app = typer.Typer()

console = Console()


@app.command()
def extract(
    zip_file: Path = typer.Option(..., "--file", "-f", help="Path to the zip file"),
    output_folder: Path = typer.Option(..., "--output-folder", "-o", help="Output folder for the dataset"),
):
    if not zip_file.is_file():
        console.print(f"File not found: {zip_file}", style="bold red")
        raise typer.Exit(code=1)

    if output_folder.is_dir():
        if any(output_folder.iterdir()):
            console.print(f"Output folder is not empty: {output_folder}", style="bold red")
            raise typer.Exit(code=1)

    output_folder.mkdir(parents=True, exist_ok=True)
    extract_zip(zip_file, output_folder)
    video_to_frames(output_folder)


def extract_zip(zip_path: Path, extract_path: Path):
    console.print(f"Extracting {zip_path} to {extract_path.resolve()}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        total_size = sum(info.file_size for info in zip_ref.infolist())

        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Extracting") as pbar:
            for info in zip_ref.infolist():
                zip_ref.extract(info, path=extract_path)
                pbar.update(info.file_size)
    console.print(f"Extraction complete: {zip_path} to {extract_path}")


def video_to_frames(source_dir: Path):
    console.print("Searching for MP4 files...")
    video_paths = list(source_dir.rglob("*.mp4"))
    if not video_paths:
        console.print("No MP4 files found.", style="bold red")
        raise typer.Exit(code=1)

    console.print(f"Found {len(video_paths)} video(s).")
    total_frames = get_total_frames(video_paths)

    console.print(f"Total frames to extract: {total_frames}")
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        n_cores = min(32, os.cpu_count() - 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_cores) as executor:
            futures = [
                executor.submit(
                    extract_all_frames,
                    video_path,
                    video_path.parent / video_path.stem,
                    pbar,
                )
                for video_path in video_paths
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    console.print("Extraction complete.")


def get_total_frames(video_paths: list[Path]) -> int:
    total = 0
    for path in tqdm(video_paths, desc="Counting frames"):
        cap = cv2.VideoCapture(str(path))
        total += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    return total


def extract_all_frames(video_path: Path, output_dir: Path, pbar: tqdm):
    cap = cv2.VideoCapture(str(video_path))
    success, frame = cap.read()
    frame_idx = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    while success:
        frame_path = output_dir / f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        pbar.update(1)
        frame_idx += 1
        success, frame = cap.read()

    cap.release()
    video_path.unlink()
    return frame_idx


if __name__ == "__main__":
    app()

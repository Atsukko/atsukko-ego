from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from tqdm.rich import tqdm

root_folder = Path(__file__).parent.parent.parent.parent.resolve()
link = "https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.XARMQA"
console = Console()

app = typer.Typer()


@app.command()
def dataset(
    output_path: Path = typer.Option(..., "--output-path", "-o", help="Output folder for the dataset"),
    resolution: int = typer.Option(256, "--resolution", "-r", help="Resolution: 256 or 384"),
):
    """
    Download the dataset from the provided link and save it to the specified output folder.
    """
    sanity_check_env()
    if resolution not in [256, 384]:
        raise typer.BadParameter("Resolution must be either 256 or 384")

    output_path = Path(output_path).expanduser().resolve()
    if output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / f"dataset_{resolution}x{resolution}.zip"

    assert output_path.suffix == ".zip", "Output path must be a directory or a zip file"

    # Check if the file already exists
    if output_path.exists():
        console.print(f"✅ File already exists: [green]{output_path}[/green]")
        return

    console.print("Starting download...", style="bold cyan")

    try:
        target_file = f"frame_v002_{resolution}.zip"
        download_file_from_edmond(output_path, target_file)
    except Exception as e:
        console.print(f"❌ An error occurred: {e}", style="bold red")
        console.print(f"You can download the dataset manually from: {link}", style="bold yellow")
        return


@app.command()
def models(
    output_path: Path = typer.Option(None, "--output-path", "-o", help="Output folder for the dataset"),
):
    """
    Download the dataset from the provided link and save it to the specified output folder.
    """
    sanity_check_env()
    if output_path is not None:
        warning = "Output path is not None, this means the models might not be saved in a place where they might not be found automatically. Run this command without the --output-path to avoid this."
        console.print(f"⚠️ {warning}", style="bold yellow")
    else:
        output_path = root_folder

    ckpt_folder = output_path / "checkpoints"

    console.print("Starting download of checkpoints...", style="bold cyan")
    backbone_folder = ckpt_folder / "backbone"
    backbone_folder.mkdir(parents=True, exist_ok=True)
    target_file = "backbone.ckpt"
    download_file_from_edmond(backbone_folder / "checkpoint.ckpt", target_file)

    stf_folder = ckpt_folder / "stf"
    stf_folder.mkdir(parents=True, exist_ok=True)
    target_file = "stf.ckpt"
    download_file_from_edmond(stf_folder / "checkpoint.ckpt", target_file)


def download_file_from_edmond(output_path: Path, target_file: str):
    """
    Download the dataset with the specified resolution.

    Args:
        output_path: Path where the dataset will be saved
        resolution: Resolution of the dataset (256 or 384)
    """
    import requests
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=False)
        page = context.new_page()
        page.goto(link)

        # Look for the file with the specified resolution
        console.print(f"Loading page and looking for {target_file} ...", style="bold cyan")

        # Wait for the files table to load
        page.wait_for_selector("table[role='grid']")

        # Find the row containing our target file
        # Using a more specific selector to find the exact row with our filename
        row_selector = f"tr.ui-widget-content:has(.fileNameOriginal:has-text('{target_file}'))"

        if not page.query_selector(row_selector):
            console.print(f"❌ File {target_file} not found", style="bold red")
            browser.close()
            return

        # Click on the access file dropdown button for our target file
        console.print(f"Found {target_file}, accessing download options...", style="bold cyan")
        page.click(f"{row_selector} a.btn-access-file")

        # Wait for and click the ZIP Archive option in the dropdown
        text = "ZIP Archive" if target_file.endswith(".zip") else "Original File Format"
        zip_option_selector = f"{row_selector} li a.btn-download:has-text('{text}')"
        page.wait_for_selector(zip_option_selector)
        page.click(zip_option_selector)

        # Wait for license popup
        console.print("Waiting for license popup...", style="bold cyan")
        page.wait_for_selector("div.form-horizontal.terms", timeout=5000)
        license_html = page.inner_html("div.form-horizontal.terms")
        license_text, terms_text = extract_license_sections(license_html)

        # Step-by-step prompts
        if not prompt_and_confirm("License/Data Use Agreement", license_text):
            console.print("❌ License not accepted. Aborting.", style="bold red")
            browser.close()
            return

        if not prompt_and_confirm("Terms of Use", terms_text):
            console.print("❌ Terms not accepted. Aborting.", style="bold red")
            browser.close()
            return

        # Click the "Accept" button
        console.print("Clicking the 'Accept' button...", style="bold green")
        page.get_by_role("button", name="Accept").click()

        # Wait for download trigger
        with page.expect_download() as download_info:
            pass

        download = download_info.value
        url = download.url

        # Extract cookies
        cookies = context.cookies()
        cookies_dict = {c["name"]: c["value"] for c in cookies}

        browser.close()

    # Manual download with tqdm
    console.print(f"Downloading {target_file}...", style="bold cyan")
    with requests.get(url, cookies=cookies_dict, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with (
            output_path.open("wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as bar,
        ):
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    console.print(f"✅ Download complete: [green]{output_path}[/green]")
    return output_path


def extract_license_sections(html: str) -> tuple[str, str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    license_text = ""
    terms_text = ""

    license_section = soup.find("label", string=lambda x: "License" in x)
    if license_section:
        license_text = license_section.find_next("div").get_text(strip=True, separator="\n")
        license_text = license_text.split("Custom Dataset Term")[0].strip()

    terms_section = soup.find("label", string=lambda x: "Terms of Use" in x)
    if terms_section:
        terms_text = terms_section.find_next("span").get_text(strip=True, separator="\n")

    return license_text, terms_text


def prompt_and_confirm(title: str, text: str) -> bool:
    console.print(Panel(Markdown(text), title=f"📄 {title}", expand=True))
    return Confirm.ask(f"[bold yellow]Do you accept the {title}?[/bold yellow]")


def sanity_check_env():
    """
    Check if the required environment is set up correctly.
    """
    try:
        import playwright
        import requests
        import tqdm
        from bs4 import BeautifulSoup
    except ImportError as e:
        console.print(f"❌ Missing package: {e.name}. Please install `framevision[all]`", style="bold red")
        return False
    return True

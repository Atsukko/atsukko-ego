import warnings

import typer
from tqdm.rich import TqdmExperimentalWarning

from .scripts.download import app as download_app
from .scripts.extract import app as extract_app

# Suppress specific warnings from tqdm
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

app = typer.Typer()
app.add_typer(download_app, name="download", help="Download data samples")
app.add_typer(extract_app)

if __name__ == "__main__":
    # Run the app
    app()

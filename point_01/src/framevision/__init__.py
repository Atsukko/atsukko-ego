import os

if os.environ.get("ENABLE_BEARTYPE") == "1":
    # Enabling runtime type checking for this package
    from beartype import BeartypeConf
    from beartype.claw import beartype_this_package

    beartype_this_package()
    BeartypeConf.is_color = False

from importlib.metadata import version

from . import autoloading, camera, dataset, geometry, loss, metrics, model, pl_wrappers, processing

__version__ = version("framevision")

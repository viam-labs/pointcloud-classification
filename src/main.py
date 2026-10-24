import asyncio

from viam.module.module import Module

# Imported for its side effect: defining Classifier registers the model with
# the registry that run_from_registry serves.
try:
    from models.classifier import Classifier  # noqa: F401
except ModuleNotFoundError:
    # when running as local module with run.sh
    from .models.classifier import Classifier  # noqa: F401


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())

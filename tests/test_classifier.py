import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from viam.proto.service.vision import Classification
from viam.services.mlmodel import Metadata, TensorInfo

# Import will work after we add helper methods
# from src.models.classifier import Classifier

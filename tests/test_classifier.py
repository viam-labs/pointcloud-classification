import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from viam.proto.service.vision import Classification
from viam.services.mlmodel import Metadata, TensorInfo

from src.models.classifier import Classifier


def test_parse_metadata_no_batch_dim():
    """Test parsing metadata with shape [N, F]"""
    # Create mock metadata
    mock_input = MagicMock()
    mock_input.name = "points"
    mock_input.shape = [1024, 3]
    mock_input.data_type = "float32"

    mock_output = MagicMock()
    mock_output.name = "logits"
    mock_output.shape = [10]
    mock_output.associated_files = []

    mock_metadata = MagicMock()
    mock_metadata.input_info = [mock_input]
    mock_metadata.output_info = [mock_output]

    classifier = MagicMock(spec=Classifier)
    result = Classifier._parse_metadata(classifier, mock_metadata)

    input_name, target_points, target_features, has_batch_dim, output_name, class_names = result

    assert input_name == "points"
    assert target_points == 1024
    assert target_features == 3
    assert has_batch_dim == False
    assert output_name == "logits"
    assert class_names is None


def test_parse_metadata_with_batch_dim():
    """Test parsing metadata with shape [1, N, F]"""
    mock_input = MagicMock()
    mock_input.name = "input_points"
    mock_input.shape = [1, 2048, 6]
    mock_input.data_type = "float32"

    mock_output = MagicMock()
    mock_output.name = "output_logits"
    mock_output.shape = [1, 5]
    mock_output.associated_files = []

    mock_metadata = MagicMock()
    mock_metadata.input_info = [mock_input]
    mock_metadata.output_info = [mock_output]

    classifier = MagicMock(spec=Classifier)
    result = Classifier._parse_metadata(classifier, mock_metadata)

    input_name, target_points, target_features, has_batch_dim, output_name, class_names = result

    assert input_name == "input_points"
    assert target_points == 2048
    assert target_features == 6
    assert has_batch_dim == True
    assert output_name == "output_logits"


def test_parse_metadata_dynamic_shape_raises():
    """Test that dynamic shapes raise ValueError"""
    mock_input = MagicMock()
    mock_input.name = "points"
    mock_input.shape = [-1, 3]
    mock_input.data_type = "float32"

    mock_metadata = MagicMock()
    mock_metadata.input_info = [mock_input]

    classifier = MagicMock(spec=Classifier)

    with pytest.raises(ValueError, match="dynamic input shape"):
        Classifier._parse_metadata(classifier, mock_metadata)

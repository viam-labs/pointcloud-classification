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


def test_logits_to_classifications_with_labels():
    """Test converting logits to classifications with class labels"""
    logits = np.array([2.0, 1.0, 3.0, 0.5, 0.1])
    class_names = ["cat", "dog", "bird", "fish", "lizard"]
    count = 3

    classifier = MagicMock(spec=Classifier)
    result = Classifier._logits_to_classifications(classifier, logits, class_names, count)

    assert len(result) == 3
    assert result[0].class_name == "bird"  # Highest logit (3.0)
    assert result[1].class_name == "cat"   # Second highest (2.0)
    assert result[2].class_name == "dog"   # Third highest (1.0)

    # Check confidences are valid probabilities
    assert 0 <= result[0].confidence <= 1
    assert 0 <= result[1].confidence <= 1
    assert 0 <= result[2].confidence <= 1

    # Check highest confidence corresponds to highest logit
    assert result[0].confidence > result[1].confidence > result[2].confidence


def test_logits_to_classifications_no_labels():
    """Test converting logits without class labels (use indices)"""
    logits = np.array([1.0, 3.0, 2.0])
    class_names = None
    count = 2

    classifier = MagicMock(spec=Classifier)
    result = Classifier._logits_to_classifications(classifier, logits, class_names, count)

    assert len(result) == 2
    assert result[0].class_name == "1"  # Index of highest logit
    assert result[1].class_name == "2"  # Index of second highest


def test_normalize_point_cloud():
    """Test point cloud normalization to unit sphere"""
    # Create test points: cube from -10 to 10
    points = np.array([
        [10.0, 10.0, 10.0],
        [-10.0, -10.0, -10.0],
        [5.0, 5.0, 5.0],
        [0.0, 0.0, 0.0]
    ])

    classifier = MagicMock(spec=Classifier)
    result = Classifier._normalize_point_cloud(classifier, points)

    # Check centered (mean near zero)
    assert np.allclose(result.mean(axis=0), [0, 0, 0], atol=1e-10)

    # Check scaled (max distance from center is 1)
    distances = np.sqrt((result ** 2).sum(axis=1))
    assert np.allclose(distances.max(), 1.0, atol=1e-6)

    # Check shape unchanged
    assert result.shape == points.shape

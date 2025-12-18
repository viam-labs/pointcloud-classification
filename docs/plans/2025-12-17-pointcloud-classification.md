# Point Cloud Classification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement whole-cloud classification using Camera point cloud data and MLModel inference to return Classification objects.

**Architecture:** Get point cloud from Camera, parse with Open3D, preprocess (sample + normalize), infer with MLModel, convert logits to Classifications via softmax.

**Tech Stack:** Python 3.11+, Viam SDK, Open3D, NumPy, pytest

---

## Task 1: Setup Test Infrastructure

**Files:**
- Create: `tests/test_classifier.py`
- Create: `tests/__init__.py`

**Step 1: Create test directory structure**

Run:
```bash
mkdir -p tests
touch tests/__init__.py
```

**Step 2: Create initial test file with imports**

Create `tests/test_classifier.py`:
```python
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from viam.proto.service.vision import Classification
from viam.services.mlmodel import Metadata, TensorInfo

# Import will work after we add helper methods
# from src.models.classifier import Classifier
```

**Step 3: Add pytest to dependencies**

Run:
```bash
uv pip install pytest pytest-asyncio
```

**Step 4: Verify pytest works**

Run: `pytest tests/ -v`
Expected: "collected 0 items" (no tests yet, but pytest works)

**Step 5: Commit**

```bash
git add tests/__init__.py tests/test_classifier.py
git commit -m "test: setup test infrastructure for classifier"
```

---

## Task 2: Helper - Parse Metadata

**Files:**
- Modify: `src/models/classifier.py`
- Test: `tests/test_classifier.py`

**Step 1: Write the failing test**

Add to `tests/test_classifier.py`:
```python
from src.models.classifier import Classifier

def test_parse_metadata_no_batch_dim():
    """Test parsing metadata with shape [N, F]"""
    # Create mock metadata
    mock_metadata = MagicMock()
    mock_metadata.input_info = [
        MagicMock(
            name="points",
            shape=[1024, 3],
            data_type="float32"
        )
    ]
    mock_metadata.output_info = [
        MagicMock(
            name="logits",
            shape=[10],
            associated_files=[]
        )
    ]

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
    mock_metadata = MagicMock()
    mock_metadata.input_info = [
        MagicMock(
            name="input_points",
            shape=[1, 2048, 6],
            data_type="float32"
        )
    ]
    mock_metadata.output_info = [
        MagicMock(
            name="output_logits",
            shape=[1, 5],
            associated_files=[]
        )
    ]

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
    mock_metadata = MagicMock()
    mock_metadata.input_info = [
        MagicMock(
            name="points",
            shape=[-1, 3],
            data_type="float32"
        )
    ]

    classifier = MagicMock(spec=Classifier)

    with pytest.raises(ValueError, match="dynamic input shape"):
        Classifier._parse_metadata(classifier, mock_metadata)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_classifier.py::test_parse_metadata_no_batch_dim -v`
Expected: FAIL - AttributeError: '_parse_metadata' not found

**Step 3: Write minimal implementation**

Add to `src/models/classifier.py` after line 95 (after reconfigure method):
```python
    def _parse_metadata(self, metadata):
        """
        Parse metadata to extract model requirements.

        Args:
            metadata: MLModel metadata object

        Returns:
            Tuple of (input_name, target_points, target_features, has_batch_dim,
                     output_name, class_names)

        Raises:
            ValueError: If shape is dynamic or unexpected format
        """
        # Get first input tensor info
        input_info = metadata.input_info[0]
        input_name = input_info.name
        shape = list(input_info.shape)

        # Parse shape: [N, F] or [1, N, F]
        if len(shape) == 2:
            has_batch_dim = False
            target_points, target_features = shape
        elif len(shape) == 3 and shape[0] == 1:
            has_batch_dim = True
            target_points, target_features = shape[1], shape[2]
        else:
            raise ValueError(f"Unexpected input shape: {shape}")

        # Check for dynamic shapes
        if target_points == -1 or target_features == -1:
            raise ValueError(
                f"Model has dynamic input shape {shape}; cannot determine requirements"
            )

        # Get output info
        output_info = metadata.output_info[0]
        output_name = output_info.name

        # Try to load class labels from associated_files
        class_names = None
        if hasattr(output_info, 'associated_files') and output_info.associated_files:
            # TODO: Parse label file in future enhancement
            # For now, just set to None
            pass

        return (
            input_name,
            int(target_points),
            int(target_features),
            has_batch_dim,
            output_name,
            class_names,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_classifier.py::test_parse_metadata -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/models/classifier.py tests/test_classifier.py
git commit -m "feat: add metadata parsing for point cloud models"
```

---

## Task 3: Helper - Logits to Classifications

**Files:**
- Modify: `src/models/classifier.py`
- Test: `tests/test_classifier.py`

**Step 1: Write the failing test**

Add to `tests/test_classifier.py`:
```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_classifier.py::test_logits_to_classifications -v`
Expected: FAIL - AttributeError: '_logits_to_classifications' not found

**Step 3: Write minimal implementation**

Add to `src/models/classifier.py` after `_parse_metadata`:
```python
    def _logits_to_classifications(
        self, logits: "np.ndarray", class_names: Optional[List[str]], count: int
    ) -> List[Classification]:
        """
        Convert model logits to Classification objects.

        Args:
            logits: Raw model output (1D array)
            class_names: List of class names (or None to use indices)
            count: Number of top classifications to return

        Returns:
            List[Classification] sorted by confidence descending
        """
        import numpy as np

        # Apply softmax: probabilities = exp(logits) / sum(exp(logits))
        # Subtract max for numerical stability
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # Sort by probability descending
        indices = np.argsort(probs)[::-1]

        # Take top count items
        top_indices = indices[:count]

        # Create Classification objects
        if class_names is None:
            class_names = [str(i) for i in range(len(logits))]

        return [
            Classification(class_name=class_names[i], confidence=float(probs[i]))
            for i in top_indices
        ]
```

Add import at top of file (after line 1):
```python
import numpy as np
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_classifier.py::test_logits_to_classifications -v`
Expected: Both tests PASS

**Step 5: Commit**

```bash
git add src/models/classifier.py tests/test_classifier.py
git commit -m "feat: add logits to classifications conversion with softmax"
```

---

## Task 4: Helper - Point Cloud Normalization

**Files:**
- Modify: `src/models/classifier.py`
- Test: `tests/test_classifier.py`

**Step 1: Write the failing test**

Add to `tests/test_classifier.py`:
```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_classifier.py::test_normalize_point_cloud -v`
Expected: FAIL - AttributeError: '_normalize_point_cloud' not found

**Step 3: Write minimal implementation**

Add to `src/models/classifier.py` after `_logits_to_classifications`:
```python
    def _normalize_point_cloud(self, points: "np.ndarray") -> "np.ndarray":
        """
        Normalize point cloud to unit sphere.

        Args:
            points: Nx3 array of XYZ coordinates

        Returns:
            Normalized points centered at origin, scaled to unit sphere
        """
        import numpy as np

        # Center at origin
        centered = points - points.mean(axis=0)

        # Scale to unit sphere
        max_dist = np.abs(centered).max()
        if max_dist > 0:
            normalized = centered / max_dist
        else:
            normalized = centered

        return normalized
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_classifier.py::test_normalize_point_cloud -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models/classifier.py tests/test_classifier.py
git commit -m "feat: add point cloud normalization to unit sphere"
```

---

## Task 5: Helper - Point Cloud Sampling (Random)

**Files:**
- Modify: `src/models/classifier.py`
- Test: `tests/test_classifier.py`

**Step 1: Write the failing test**

Add to `tests/test_classifier.py`:
```python
def test_sample_point_cloud_downsample():
    """Test downsampling to fewer points"""
    points = np.random.rand(2000, 3)
    target_count = 1024

    classifier = MagicMock(spec=Classifier)
    result = Classifier._sample_point_cloud(classifier, points, target_count, "random")

    assert result.shape == (1024, 3)
    # Check all result points exist in original
    for point in result[:5]:  # Check first few
        # Should be close to one of the original points
        distances = np.sqrt(((points - point) ** 2).sum(axis=1))
        assert distances.min() < 1e-6


def test_sample_point_cloud_upsample():
    """Test upsampling to more points (with duplication)"""
    points = np.random.rand(100, 3)
    target_count = 500

    classifier = MagicMock(spec=Classifier)
    result = Classifier._sample_point_cloud(classifier, points, target_count, "random")

    assert result.shape == (500, 3)
    # All result points should match one of the original points
    for point in result[:10]:  # Check first few
        distances = np.sqrt(((points - point) ** 2).sum(axis=1))
        assert distances.min() < 1e-6


def test_sample_point_cloud_exact_count():
    """Test when point count already matches"""
    points = np.random.rand(1024, 3)
    target_count = 1024

    classifier = MagicMock(spec=Classifier)
    result = Classifier._sample_point_cloud(classifier, points, target_count, "random")

    assert result.shape == (1024, 3)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_classifier.py::test_sample_point_cloud -v`
Expected: FAIL - AttributeError: '_sample_point_cloud' not found

**Step 3: Write minimal implementation**

Add to `src/models/classifier.py` after `_normalize_point_cloud`:
```python
    def _sample_point_cloud(
        self, points: "np.ndarray", target_count: int, method: str
    ) -> "np.ndarray":
        """
        Sample point cloud to target number of points.

        Args:
            points: Nx3 (or NxF) array of point features
            target_count: Desired number of points
            method: Sampling method ("random", "voxel", or "fps")

        Returns:
            Sampled points with shape [target_count, F]
        """
        import numpy as np

        current_count = points.shape[0]

        if current_count == target_count:
            return points

        # For now, only implement random sampling
        # TODO: Add voxel and fps methods later
        if method != "random":
            self.logger.warning(
                f"Sampling method '{method}' not yet implemented, using random"
            )

        # Random sampling (works for both up and down sampling)
        indices = np.random.choice(current_count, target_count, replace=(current_count < target_count))
        return points[indices]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_classifier.py::test_sample_point_cloud -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/models/classifier.py tests/test_classifier.py
git commit -m "feat: add random point cloud sampling (up/down)"
```

---

## Task 6: Helper - Parse Point Cloud Bytes

**Files:**
- Modify: `src/models/classifier.py`
- Test: `tests/test_classifier.py`

**Step 1: Write the failing test**

Add to `tests/test_classifier.py`:
```python
import open3d as o3d
import io

def test_parse_point_cloud_pcd_format():
    """Test parsing PCD format point cloud bytes"""
    # Create a simple point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]))

    # Convert to bytes (simulate camera.get_point_cloud output)
    # Write to buffer
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pcd', delete=False) as f:
        o3d.io.write_point_cloud(f.name, pcd)
        f.seek(0)
        with open(f.name, 'rb') as pcd_file:
            pcd_bytes = pcd_file.read()

    classifier = MagicMock(spec=Classifier)
    result = Classifier._parse_point_cloud(classifier, pcd_bytes, "application/pcd")

    assert isinstance(result, o3d.geometry.PointCloud)
    points = np.asarray(result.points)
    assert points.shape == (3, 3)
    assert np.allclose(points[0], [1.0, 2.0, 3.0])
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_classifier.py::test_parse_point_cloud_pcd_format -v`
Expected: FAIL - AttributeError: '_parse_point_cloud' not found

**Step 3: Write minimal implementation**

Add to `src/models/classifier.py` after `_sample_point_cloud`:
```python
    def _parse_point_cloud(self, pcd_bytes: bytes, mimetype: str) -> "o3d.geometry.PointCloud":
        """
        Parse point cloud bytes into Open3D PointCloud object.

        Args:
            pcd_bytes: Raw point cloud bytes from camera
            mimetype: MIME type of the point cloud data

        Returns:
            Open3D PointCloud object

        Raises:
            RuntimeError: If parsing fails
        """
        import open3d as o3d
        import tempfile
        import os

        try:
            # Write bytes to temporary file
            # Open3D requires file path, not bytes directly
            with tempfile.NamedTemporaryFile(suffix='.pcd', delete=False) as tmp_file:
                tmp_file.write(pcd_bytes)
                tmp_path = tmp_file.name

            # Read point cloud from file
            pcd = o3d.io.read_point_cloud(tmp_path)

            # Clean up temp file
            os.unlink(tmp_path)

            if len(pcd.points) == 0:
                raise RuntimeError("Parsed point cloud is empty")

            return pcd

        except Exception as e:
            raise RuntimeError(f"Failed to parse point cloud data: {e}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_classifier.py::test_parse_point_cloud_pcd_format -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models/classifier.py tests/test_classifier.py
git commit -m "feat: add point cloud bytes parsing with Open3D"
```

---

## Task 7: Helper - Preprocess Point Cloud

**Files:**
- Modify: `src/models/classifier.py`
- Test: `tests/test_classifier.py`

**Step 1: Write the failing test**

Add to `tests/test_classifier.py`:
```python
def test_preprocess_point_cloud_xyz_only():
    """Test preprocessing with XYZ features only"""
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(2000, 3) * 10)

    target_points = 1024
    target_features = 3

    classifier = MagicMock(spec=Classifier)
    classifier._sample_point_cloud = Classifier._sample_point_cloud.__get__(classifier, Classifier)
    classifier._normalize_point_cloud = Classifier._normalize_point_cloud.__get__(classifier, Classifier)

    result = Classifier._preprocess_point_cloud(
        classifier, pcd, target_points, target_features, "random"
    )

    assert result.shape == (1024, 3)
    # Check normalized (max value should be ~1)
    assert np.abs(result).max() <= 1.0 + 1e-6


def test_preprocess_point_cloud_xyz_rgb():
    """Test preprocessing with XYZ+RGB features"""
    # Create Open3D point cloud with colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(500, 3) * 10)
    pcd.colors = o3d.utility.Vector3dVector(np.random.rand(500, 3))

    target_points = 256
    target_features = 6

    classifier = MagicMock(spec=Classifier)
    classifier._sample_point_cloud = Classifier._sample_point_cloud.__get__(classifier, Classifier)
    classifier._normalize_point_cloud = Classifier._normalize_point_cloud.__get__(classifier, Classifier)

    result = Classifier._preprocess_point_cloud(
        classifier, pcd, target_points, target_features, "random"
    )

    assert result.shape == (256, 6)
    # XYZ should be normalized
    assert np.abs(result[:, :3]).max() <= 1.0 + 1e-6
    # RGB should be in [0, 1]
    assert result[:, 3:].min() >= 0.0
    assert result[:, 3:].max() <= 1.0


def test_preprocess_point_cloud_missing_rgb():
    """Test error when RGB required but not available"""
    # Create point cloud without colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))

    classifier = MagicMock(spec=Classifier)

    with pytest.raises(ValueError, match="requires RGB"):
        Classifier._preprocess_point_cloud(
            classifier, pcd, 100, 6, "random"
        )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_classifier.py::test_preprocess_point_cloud -v`
Expected: FAIL - AttributeError: '_preprocess_point_cloud' not found

**Step 3: Write minimal implementation**

Add to `src/models/classifier.py` after `_parse_point_cloud`:
```python
    def _preprocess_point_cloud(
        self,
        cloud: "o3d.geometry.PointCloud",
        target_points: int,
        target_features: int,
        sampling_method: str,
    ) -> "np.ndarray":
        """
        Preprocess point cloud for model inference.

        Args:
            cloud: Open3D point cloud object
            target_points: Number of points required by model (N)
            target_features: Number of features per point (3, 6, or 9)
            sampling_method: "random", "voxel", or "fps"

        Returns:
            numpy array of shape [target_points, target_features]

        Raises:
            ValueError: If required features are missing from cloud
        """
        import numpy as np

        # Extract XYZ (always present)
        points = np.asarray(cloud.points)

        # Check and extract additional features
        features = [points]

        # Extract RGB if needed (target_features >= 6)
        if target_features >= 6:
            if not cloud.has_colors():
                raise ValueError(
                    f"Model requires RGB data (shape [N,{target_features}]) "
                    "but point cloud has no colors"
                )
            colors = np.asarray(cloud.colors)
            features.append(colors)

        # Extract normals if needed (target_features >= 9)
        if target_features >= 9:
            if not cloud.has_normals():
                raise ValueError(
                    f"Model requires normals (shape [N,{target_features}]) "
                    "but point cloud has none"
                )
            normals = np.asarray(cloud.normals)
            features.append(normals)

        # Concatenate features
        combined = np.concatenate(features, axis=1)

        # Resample to target_points
        sampled = self._sample_point_cloud(combined, target_points, sampling_method)

        # Normalize XYZ coordinates only (first 3 columns)
        xyz_normalized = self._normalize_point_cloud(sampled[:, :3])

        # Combine normalized XYZ with other features
        if target_features == 3:
            result = xyz_normalized
        else:
            # Keep RGB/normals as-is, replace XYZ with normalized
            result = np.concatenate([xyz_normalized, sampled[:, 3:]], axis=1)

        return result
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_classifier.py::test_preprocess_point_cloud -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/models/classifier.py tests/test_classifier.py
git commit -m "feat: add full point cloud preprocessing pipeline"
```

---

## Task 8: Implement get_classifications_from_camera

**Files:**
- Modify: `src/models/classifier.py:146-155`
- Test: `tests/test_classifier.py`

**Step 1: Write the failing test**

Add to `tests/test_classifier.py`:
```python
@pytest.mark.asyncio
async def test_get_classifications_from_camera_integration():
    """Integration test for get_classifications_from_camera"""
    # Setup mocks
    mock_camera = AsyncMock()
    mock_mlmodel = AsyncMock()

    # Create test point cloud bytes
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(2000, 3) * 10)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pcd', delete=False) as f:
        o3d.io.write_point_cloud(f.name, pcd)
        with open(f.name, 'rb') as pcd_file:
            pcd_bytes = pcd_file.read()

    mock_camera.get_point_cloud = AsyncMock(return_value=(pcd_bytes, "application/pcd"))

    # Mock metadata
    mock_metadata = MagicMock()
    mock_metadata.input_info = [
        MagicMock(name="points", shape=[1024, 3])
    ]
    mock_metadata.output_info = [
        MagicMock(name="logits", shape=[3], associated_files=[])
    ]
    mock_mlmodel.metadata = AsyncMock(return_value=mock_metadata)

    # Mock inference output
    output_logits = np.array([2.0, 1.0, 3.0])
    mock_mlmodel.infer = AsyncMock(return_value={"logits": output_logits})

    # Create classifier instance
    from viam.proto.app.robot import ComponentConfig
    from google.protobuf.struct_pb2 import Struct

    config = ComponentConfig()
    config.name = "test_classifier"
    attrs = Struct()
    attrs["mlmodel_name"] = "test_model"
    attrs["camera_name"] = "test_camera"
    config.attributes.CopyFrom(attrs)

    # Create mock dependencies
    from viam.components.camera import Camera
    from viam.services.mlmodel import MLModel

    dependencies = {
        Camera.get_resource_name("test_camera"): mock_camera,
        MLModel.get_resource_name("test_model"): mock_mlmodel,
    }

    classifier = Classifier.new(config, dependencies)

    # Test
    results = await classifier.get_classifications_from_camera("test_camera", count=2)

    assert len(results) == 2
    assert results[0].class_name == "2"  # Highest logit
    assert results[1].class_name == "0"  # Second highest
    assert results[0].confidence > results[1].confidence
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_classifier.py::test_get_classifications_from_camera_integration -v`
Expected: FAIL - NotImplementedError

**Step 3: Write implementation**

Replace `src/models/classifier.py` lines 146-155 with:
```python
    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        """Get classifications from point cloud captured by camera."""
        # Get metadata
        metadata = await self.mlmodel.metadata()
        (
            input_name,
            target_points,
            target_features,
            has_batch_dim,
            output_name,
            class_names,
        ) = self._parse_metadata(metadata)

        # Get camera
        if camera_name == "" and self.default_camera == "":
            raise ValueError(
                "No camera name provided and no default camera name configured"
            )
        elif camera_name == "":
            camera_name = self.default_camera
        camera = self.getCamera(camera_name)

        # Get point cloud
        pcd_bytes, mimetype = await camera.get_point_cloud(
            extra=extra, timeout=timeout
        )

        # Parse with Open3D
        cloud = self._parse_point_cloud(pcd_bytes, mimetype)

        # Preprocess
        sampling_method = getattr(self, "sampling_method", "random")
        preprocessed = self._preprocess_point_cloud(
            cloud, target_points, target_features, sampling_method
        )

        # Add batch dimension if needed
        if has_batch_dim:
            preprocessed = preprocessed[np.newaxis, ...]

        # Inference
        input_tensors = {input_name: preprocessed}
        output_tensors = await self.mlmodel.infer(
            input_tensors, extra=extra, timeout=timeout
        )

        # Extract output
        logits = output_tensors[output_name]
        if has_batch_dim:
            logits = logits[0]  # Remove batch dimension

        # Convert to classifications
        return self._logits_to_classifications(logits, class_names, count)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_classifier.py::test_get_classifications_from_camera_integration -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models/classifier.py tests/test_classifier.py
git commit -m "feat: implement get_classifications_from_camera"
```

---

## Task 9: Update capture_all_from_camera for Classifications

**Files:**
- Modify: `src/models/classifier.py:97-124`
- Test: `tests/test_classifier.py`

**Step 1: Write the failing test**

Add to `tests/test_classifier.py`:
```python
@pytest.mark.asyncio
async def test_capture_all_from_camera_with_classifications():
    """Test capture_all_from_camera returns classifications"""
    # Setup mocks (similar to previous test)
    mock_camera = AsyncMock()
    mock_mlmodel = AsyncMock()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(1000, 3) * 5)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pcd', delete=False) as f:
        o3d.io.write_point_cloud(f.name, pcd)
        with open(f.name, 'rb') as pcd_file:
            pcd_bytes = pcd_file.read()

    mock_camera.get_point_cloud = AsyncMock(return_value=(pcd_bytes, "application/pcd"))

    mock_metadata = MagicMock()
    mock_metadata.input_info = [MagicMock(name="points", shape=[512, 3])]
    mock_metadata.output_info = [MagicMock(name="logits", shape=[5], associated_files=[])]
    mock_mlmodel.metadata = AsyncMock(return_value=mock_metadata)
    mock_mlmodel.infer = AsyncMock(return_value={"logits": np.array([1, 2, 3, 4, 5])})

    from viam.proto.app.robot import ComponentConfig
    from google.protobuf.struct_pb2 import Struct
    from viam.components.camera import Camera
    from viam.services.mlmodel import MLModel

    config = ComponentConfig()
    config.name = "test"
    attrs = Struct()
    attrs["mlmodel_name"] = "model"
    attrs["camera_name"] = "camera"
    config.attributes.CopyFrom(attrs)

    dependencies = {
        Camera.get_resource_name("camera"): mock_camera,
        MLModel.get_resource_name("model"): mock_mlmodel,
    }

    classifier = Classifier.new(config, dependencies)

    # Test with return_classifications=True
    result = await classifier.capture_all_from_camera(
        "camera", return_classifications=True
    )

    assert result.classifications is not None
    assert len(result.classifications) > 0
    assert result.classifications[0].class_name == "4"  # Highest logit (5.0 at index 4)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_classifier.py::test_capture_all_from_camera_with_classifications -v`
Expected: FAIL - result.classifications is None

**Step 3: Write implementation**

Replace `src/models/classifier.py` lines 97-124 with:
```python
    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        result = CaptureAllResult()

        if camera_name == "" and self.default_camera == "":
            raise ValueError(
                "No camera name provided and no default camera name configured"
            )
        elif camera_name == "":
            camera_name = self.default_camera
        camera = self.getCamera(camera_name)

        if return_image:
            images, _ = await camera.get_images(timeout=timeout)

            if len(images) > 0:
                result.image = images[0]

        if return_classifications:
            # Get count from extra or use default
            count = 5  # Default count
            if extra and "count" in extra:
                count = int(extra["count"])

            classifications = await self.get_classifications_from_camera(
                camera_name, count, extra=extra, timeout=timeout
            )
            result.classifications = classifications

        return result
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_classifier.py::test_capture_all_from_camera_with_classifications -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models/classifier.py tests/test_classifier.py
git commit -m "feat: add classification support to capture_all_from_camera"
```

---

## Task 10: Update get_classifications to Raise Clear Error

**Files:**
- Modify: `src/models/classifier.py:157-166`

**Step 1: Write the failing test**

Add to `tests/test_classifier.py`:
```python
@pytest.mark.asyncio
async def test_get_classifications_raises_not_implemented():
    """Test get_classifications raises helpful error"""
    from viam.proto.app.robot import ComponentConfig
    from google.protobuf.struct_pb2 import Struct
    from viam.components.camera import Camera
    from viam.services.mlmodel import MLModel

    config = ComponentConfig()
    config.name = "test"
    attrs = Struct()
    attrs["mlmodel_name"] = "model"
    config.attributes.CopyFrom(attrs)

    dependencies = {
        MLModel.get_resource_name("model"): AsyncMock(),
    }

    classifier = Classifier.new(config, dependencies)

    mock_image = MagicMock()

    with pytest.raises(NotImplementedError, match="camera input"):
        await classifier.get_classifications(mock_image, count=5)
```

**Step 2: Run test to verify current behavior**

Run: `pytest tests/test_classifier.py::test_get_classifications_raises_not_implemented -v`
Expected: PASS (already raises NotImplementedError, but check message)

**Step 3: Update error message**

Replace `src/models/classifier.py` lines 157-166 with:
```python
    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        self.logger.error("Point cloud classification requires camera input")
        raise NotImplementedError(
            "get_classifications() not supported for point clouds; "
            "use get_classifications_from_camera() instead"
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_classifier.py::test_get_classifications_raises_not_implemented -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models/classifier.py tests/test_classifier.py
git commit -m "docs: improve error message for get_classifications"
```

---

## Task 11: Add sampling_method Configuration Support

**Files:**
- Modify: `src/models/classifier.py:73-95`

**Step 1: Update reconfigure to store sampling_method**

Modify `src/models/classifier.py` reconfigure method (lines 73-95):
```python
    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        attrs = struct_to_dict(config.attributes)

        def getCamera(name: str) -> Camera:
            return cast(Camera, dependencies.get(Camera.get_resource_name(name)))

        self.getCamera = getCamera
        self.default_camera = str(attrs.get("camera_name", ""))

        # Store sampling method (default: random)
        self.sampling_method = str(attrs.get("sampling_method", "random"))
        if self.sampling_method not in ["random", "voxel", "fps"]:
            self.logger.warning(
                f"Invalid sampling_method '{self.sampling_method}', using 'random'"
            )
            self.sampling_method = "random"

        try:
            self.mlmodel = cast(
                MLModel,
                dependencies.get(
                    MLModel.get_resource_name(str(attrs.get("mlmodel_name")))
                ),
            )
        except Exception as err:
            self.logger.error(
                f"Unable to configure pointcloud classifier vision service: {err}"
            )
            raise err
```

**Step 2: Add test for configuration**

Add to `tests/test_classifier.py`:
```python
def test_reconfigure_with_sampling_method():
    """Test that sampling_method is stored from config"""
    from viam.proto.app.robot import ComponentConfig
    from google.protobuf.struct_pb2 import Struct
    from viam.services.mlmodel import MLModel

    config = ComponentConfig()
    config.name = "test"
    attrs = Struct()
    attrs["mlmodel_name"] = "model"
    attrs["sampling_method"] = "voxel"
    config.attributes.CopyFrom(attrs)

    dependencies = {
        MLModel.get_resource_name("model"): MagicMock(),
    }

    classifier = Classifier.new(config, dependencies)

    assert classifier.sampling_method == "voxel"


def test_reconfigure_default_sampling_method():
    """Test default sampling_method is random"""
    from viam.proto.app.robot import ComponentConfig
    from google.protobuf.struct_pb2 import Struct
    from viam.services.mlmodel import MLModel

    config = ComponentConfig()
    config.name = "test"
    attrs = Struct()
    attrs["mlmodel_name"] = "model"
    config.attributes.CopyFrom(attrs)

    dependencies = {
        MLModel.get_resource_name("model"): MagicMock(),
    }

    classifier = Classifier.new(config, dependencies)

    assert classifier.sampling_method == "random"
```

**Step 3: Run test to verify it passes**

Run: `pytest tests/test_classifier.py::test_reconfigure -v`
Expected: Both tests PASS

**Step 4: Commit**

```bash
git add src/models/classifier.py tests/test_classifier.py
git commit -m "feat: add sampling_method configuration support"
```

---

## Task 12: Run Full Test Suite

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 2: Check test coverage (optional)**

Run:
```bash
uv pip install pytest-cov
pytest tests/ --cov=src/models/classifier --cov-report=term-missing
```

Review coverage and identify any gaps.

**Step 3: Fix any failing tests**

If any tests fail, fix them before proceeding.

**Step 4: Final commit**

```bash
git add -A
git commit -m "test: ensure all point cloud classification tests pass"
```

---

## Task 13: Manual Testing (Optional)

**Files:**
- Create: `examples/test_classifier_manual.py`

**Step 1: Create manual test script**

Create `examples/test_classifier_manual.py`:
```python
"""
Manual test script for point cloud classifier.
Requires a running viam-server with configured camera and mlmodel.
"""
import asyncio
from viam.robot.client import RobotClient
from viam.services.vision import Vision

async def test_classifier():
    # Connect to robot
    robot = await RobotClient.at_address("localhost:8080")

    # Get vision service
    classifier = Vision.from_robot(robot, "classifier")

    # Test get_classifications_from_camera
    results = await classifier.get_classifications_from_camera(
        camera_name="depth_camera",
        count=5
    )

    print("Classifications:")
    for i, classification in enumerate(results):
        print(f"  {i+1}. {classification.class_name}: {classification.confidence:.4f}")

    # Test capture_all_from_camera
    capture_result = await classifier.capture_all_from_camera(
        camera_name="depth_camera",
        return_classifications=True
    )

    print("\nCapture All Results:")
    if capture_result.classifications:
        for classification in capture_result.classifications:
            print(f"  {classification.class_name}: {classification.confidence:.4f}")

    await robot.close()

if __name__ == "__main__":
    asyncio.run(test_classifier())
```

**Step 2: Document manual testing in plan**

This script can be used for manual integration testing with real hardware.

---

## Summary

This plan implements point cloud classification in 13 tasks:

1. ✅ Setup test infrastructure
2. ✅ Parse metadata helper
3. ✅ Logits to classifications helper
4. ✅ Normalize point cloud helper
5. ✅ Sample point cloud helper (random only)
6. ✅ Parse point cloud bytes helper
7. ✅ Preprocess point cloud helper (combines sampling + normalization)
8. ✅ Implement get_classifications_from_camera
9. ✅ Update capture_all_from_camera
10. ✅ Update get_classifications error message
11. ✅ Add sampling_method configuration
12. ✅ Run full test suite
13. ✅ Manual testing script (optional)

**Future Enhancements:**
- Implement voxel and FPS sampling methods
- Add metadata caching optimization
- Parse class labels from associated_files
- Support for normals (9-feature input)
- Performance optimizations

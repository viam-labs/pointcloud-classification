# Point Cloud Classification Design

**Date:** 2025-12-17
**Status:** Approved

## Overview

Add support for classifying point cloud data from the Viam Camera API by processing it into tensors for the Viam MLModel Service, then returning Classification objects from `get_classifications_from_camera()` and `capture_all_from_camera()`.

**Approach:** Whole-cloud classification - the entire point cloud receives one set of classifications (top-K classes with confidence scores).

## Architecture & Data Flow

The point cloud classification pipeline:

1. **Acquisition**: Call `camera.get_point_cloud()` → receives `(bytes, mimetype)`

2. **Parsing**: Use Open3D to deserialize bytes into structured point cloud with XYZ coordinates (and optionally RGB colors, normals)

3. **Feature Extraction**: Based on model's expected input shape from `metadata()`:
   - Shape `[N, 3]` or `[1, N, 3]` → Extract XYZ only
   - Shape `[N, 6]` or `[1, N, 6]` → Extract XYZ + RGB
   - Shape `[N, 9]` or `[1, N, 9]` → Extract XYZ + RGB + normals
   - Fail fast if required features are missing from point cloud

4. **Preprocessing**:
   - Sample/resample to target point count (N extracted from metadata shape)
   - Normalize to unit sphere (center at origin, scale by max radius)
   - Format as numpy array with correct shape including batch dimension if needed

5. **Inference**: Call `mlmodel.infer(input_tensors)` → receives output tensor dictionary

6. **Post-processing**:
   - Extract logits from output tensor
   - Apply softmax to convert to probabilities
   - Load class names from metadata's associated_files (or use numeric indices)
   - Sort by confidence, return top-K as Classification objects

## Configuration

New configuration attributes (in addition to existing `mlmodel_name` and `camera_name`):

**`sampling_method`** (string, optional, default: `"random"`)
- Controls how point clouds are resampled to match model's expected point count
- Options:
  - `"random"` - Random sampling (down-sample) or random duplication (up-sample)
  - `"voxel"` - Voxel grid downsampling using Open3D (deterministic, better coverage)
  - `"fps"` - Farthest point sampling (best coverage, most expensive)
- Note: Voxel and FPS only work for down-sampling; falls back to random for up-sampling

**Example configuration:**
```json
{
  "mlmodel_name": "my-pointcloud-classifier",
  "camera_name": "depth-camera",
  "sampling_method": "random"
}
```

**Metadata-driven decisions** (not configurable):
- Point count (N) - extracted from `input_info[0].shape`
- Feature requirements - inferred from shape's feature dimension
- Batch dimension - inferred from shape (`[N, F]` vs `[1, N, F]`)
- Class labels - loaded from `output_info[0].associated_files` or use numeric indices

## Implementation Details

### Metadata Handling

- Call `await mlmodel.metadata()` at the start of `get_classifications_from_camera()` and `capture_all_from_camera()`
- Can add caching optimization later if needed
- Parse input tensor info to extract:
  - Expected shape (handle both `[N, F]` and `[1, N, F]` formats)
  - Point count N (if -1 or dynamic, raise error)
  - Feature count F to determine XYZ (3) vs XYZ+RGB (6) vs XYZ+RGB+normals (9)
- Parse output tensor info to find associated class labels file
- Load class labels from associated file or use numeric indices ("0", "1", "2"...) as fallback

### Point Cloud Preprocessing

```python
def _preprocess_point_cloud(cloud, target_points, target_features, sampling_method):
    """
    Preprocess point cloud for model inference.

    Args:
        cloud: Open3D point cloud object
        target_points: Number of points required by model (N)
        target_features: Number of features per point (3, 6, or 9)
        sampling_method: "random", "voxel", or "fps"

    Returns:
        numpy array of shape [target_points, target_features]
    """
    # Extract XYZ (always present)
    points = np.asarray(cloud.points)

    # Extract RGB if needed (target_features >= 6)
    if target_features >= 6:
        if not cloud.has_colors():
            raise ValueError("Model requires RGB data but point cloud has no colors")
        colors = np.asarray(cloud.colors)

    # Extract normals if needed (target_features >= 9)
    if target_features >= 9:
        if not cloud.has_normals():
            raise ValueError("Model requires normals but point cloud has none")
        normals = np.asarray(cloud.normals)

    # Resample to target_points using sampling_method
    # - Random: np.random.choice for down/up sampling
    # - Voxel: cloud.voxel_down_sample() then random if needed
    # - FPS: farthest point sampling then random if needed

    # Normalize to unit sphere
    # - Center: points -= points.mean(axis=0)
    # - Scale: points /= np.abs(points).max()

    # Concatenate features [XYZ] or [XYZ, RGB] or [XYZ, RGB, normals]
    # Return numpy array
```

### Output Processing

```python
def _logits_to_classifications(logits, class_names, count):
    """
    Convert model logits to Classification objects.

    Args:
        logits: Raw model output (1D array)
        class_names: List of class names (or None to use indices)
        count: Number of top classifications to return

    Returns:
        List[Classification] sorted by confidence descending
    """
    # Apply softmax: probabilities = exp(logits) / sum(exp(logits))
    probs = np.exp(logits - np.max(logits))  # subtract max for numerical stability
    probs = probs / probs.sum()

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

### Metadata Parsing

```python
def _parse_metadata(metadata):
    """
    Parse metadata to extract model requirements.

    Returns:
        (input_name, target_points, target_features, has_batch_dim,
         output_name, class_names)
    """
    # Get first input tensor info
    input_info = metadata.input_info[0]
    input_name = input_info.name
    shape = input_info.shape

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
        raise ValueError(f"Model has dynamic input shape {shape}; cannot determine requirements")

    # Get output info
    output_info = metadata.output_info[0]
    output_name = output_info.name

    # Try to load class labels from associated_files
    class_names = None
    if output_info.associated_files:
        # Parse first associated file as labels (one per line)
        # class_names = load_labels_file(output_info.associated_files[0])
        pass

    return (input_name, target_points, target_features, has_batch_dim,
            output_name, class_names)
```

## Method Implementations

### `get_classifications_from_camera(camera_name, count)`

```python
async def get_classifications_from_camera(
    self, camera_name: str, count: int, **kwargs
) -> List[Classification]:
    # Get metadata
    metadata = await self.mlmodel.metadata()
    input_name, target_points, target_features, has_batch_dim, output_name, class_names = \
        self._parse_metadata(metadata)

    # Get camera
    if camera_name == "" and self.default_camera == "":
        raise ValueError("No camera name provided and no default camera name configured")
    elif camera_name == "":
        camera_name = self.default_camera
    camera = self.getCamera(camera_name)

    # Get point cloud
    pcd_bytes, mimetype = await camera.get_point_cloud()

    # Parse with Open3D
    cloud = self._parse_point_cloud(pcd_bytes, mimetype)

    # Preprocess
    sampling_method = self.config.get("sampling_method", "random")
    preprocessed = self._preprocess_point_cloud(
        cloud, target_points, target_features, sampling_method
    )

    # Add batch dimension if needed
    if has_batch_dim:
        preprocessed = preprocessed[np.newaxis, ...]

    # Inference
    input_tensors = {input_name: preprocessed}
    output_tensors = await self.mlmodel.infer(input_tensors)

    # Extract output
    logits = output_tensors[output_name]
    if has_batch_dim:
        logits = logits[0]  # Remove batch dimension

    # Convert to classifications
    return self._logits_to_classifications(logits, class_names, count)
```

### `capture_all_from_camera(...)`

Update to support `return_classifications=True`:

```python
async def capture_all_from_camera(
    self, camera_name: str,
    return_image: bool = False,
    return_classifications: bool = False,
    return_detections: bool = False,
    return_object_point_clouds: bool = False,
    **kwargs
) -> CaptureAllResult:
    result = CaptureAllResult()

    # Existing image logic
    if return_image:
        # ... existing code ...
        pass

    # New classification logic
    if return_classifications:
        # Use default count or from kwargs
        count = kwargs.get('extra', {}).get('count', 5)
        classifications = await self.get_classifications_from_camera(
            camera_name, count, **kwargs
        )
        result.classifications = classifications

    return result
```

### `get_classifications(image, count)`

```python
async def get_classifications(
    self, image: ViamImage, count: int, **kwargs
) -> List[Classification]:
    self.logger.error("Point cloud classification requires camera input")
    raise NotImplementedError(
        "get_classifications() not supported for point clouds; "
        "use get_classifications_from_camera() instead"
    )
```

## Error Handling

**Fail-fast with clear error messages:**

**Missing features:**
- `ValueError: "Model requires RGB data (shape [N,6]) but point cloud has no colors"`
- `ValueError: "Model requires normals (shape [N,9]) but point cloud has none"`

**Metadata issues:**
- `ValueError: "Model has dynamic input shape [..., -1, ...]; cannot determine required point count"`
- `ValueError: "Unable to parse point count from input shape: {shape}"`

**Camera issues:**
- `ValueError: "No camera name provided and no default camera name configured"`
- `RuntimeError: "Failed to parse point cloud data: {error}"`

**Model output issues:**
- `ValueError: "Unexpected model output shape: {shape}"`

**Graceful fallbacks:**
- Missing class labels → Use numeric indices ("0", "1", "2", ...)
- Output has fewer than `count` classes → Return all available classes

## Helper Methods

New helper methods to implement:

- `_parse_metadata(metadata)` → extract shapes, features, class labels
- `_parse_point_cloud(bytes, mimetype)` → Open3D point cloud object
- `_preprocess_point_cloud(cloud, target_points, target_features, method)` → numpy array
- `_logits_to_classifications(logits, class_names, count)` → List[Classification]

## Testing Considerations

- Unit tests for preprocessing (sampling, normalization)
- Unit tests for softmax and top-K selection
- Integration tests with mock MLModel service
- Test cases for missing features (no RGB, no normals)
- Test cases for various input shapes (batch vs no batch)
- Test edge cases (very few points, very many points)

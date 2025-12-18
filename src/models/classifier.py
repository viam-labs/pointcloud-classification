from typing import ClassVar, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
from typing_extensions import Self
from viam.components.camera import Camera
from viam.services.mlmodel import MLModel
from viam.media.video import ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.vision import Vision, CaptureAllResult
from viam.utils import ValueTypes, struct_to_dict


class Classifier(Vision, EasyResource):
    # To enable debug-level logging, either run viam-server with the --debug option,
    # or configure your resource/machine to display debug logs.
    MODEL: ClassVar[Model] = Model(
        ModelFamily("viam-labs", "pointcloud-classification"), "classifier"
    )

    mlmodel: MLModel
    default_camera: str

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """This method creates a new instance of this Vision service.
        The default implementation sets the name from the `config` parameter.

        Args:
            config (ComponentConfig): The configuration for this resource
            dependencies (Mapping[ResourceName, ResourceBase]): The dependencies (both required and optional)

        Returns:
            Self: The resource
        """
        self = super().new(config, dependencies)
        self.reconfigure(config, dependencies)
        return self

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        """This method allows you to validate the configuration object received from the machine,
        as well as to return any required dependencies or optional dependencies based on that `config`.

        Args:
            config (ComponentConfig): The configuration for this resource

        Returns:
            Tuple[Sequence[str], Sequence[str]]: A tuple where the
                first element is a list of required dependencies and the
                second element is a list of optional dependencies
        """
        attrs = struct_to_dict(config.attributes)
        deps = []

        if mlmodel := attrs.get("mlmodel_name", None):
            deps.append(mlmodel)
        else:
            raise ValueError("mlmodel_name field is required in configuration")

        if camera := attrs.get("camera_name", None):
            deps.append(camera)

        return deps, []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        attrs = struct_to_dict(config.attributes)

        def getCamera(name: str) -> Camera:
            return cast(Camera, dependencies.get(Camera.get_resource_name(name)))

        self.getCamera = getCamera
        self.default_camera = str(attrs.get("camera_name"))

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
            images, _ = await camera.get_images()

            if len(images) > 0:
                result.image = images[0]

        return result

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        self.logger.error("`get_detections_from_camera` is not implemented")
        raise NotImplementedError()

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        self.logger.error("`get_detections` is not implemented")
        raise NotImplementedError()

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        self.logger.error("`get_classifications_from_camera` is not implemented")
        raise NotImplementedError()

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        self.logger.error("`get_classifications` is not implemented")
        raise NotImplementedError()

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[PointCloudObject]:
        self.logger.error("`get_object_point_clouds` is not implemented")
        raise NotImplementedError()

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> Vision.Properties:
        properties = Vision.Properties(
            classifications_supported=True,
            detections_supported=False,
            object_point_clouds_supported=False,
        )
        return properties

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        self.logger.error("`do_command` is not implemented")
        raise NotImplementedError()

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Viam module that implements a vision service for point cloud classification. The module is registered as `viam-labs:pointcloud-classification:classifier` and provides a Vision service that processes point cloud data using ML models.

## Build System and Package Manager

This project uses **uv** (a fast Python package installer and resolver) rather than pip or poetry. All dependency management and virtual environment operations should use uv.

## Development Commands

### Setup
```bash
./setup.sh
```
This installs uv (if needed), creates a virtual environment, syncs dependencies from pyproject.toml, and installs PyInstaller for building.

### Running Locally
```bash
./run.sh
```
Runs the module locally using the virtual environment. The module will be available through the Viam SDK module registry.

### Building for Distribution
```bash
./build.sh
```
Uses PyInstaller to create a standalone executable with all dependencies bundled, then packages it as `dist/archive.tar.gz` for Viam module deployment. The build includes special handling for vosk and pyaudio dependencies.

## Architecture

### Module Structure

- **src/main.py**: Entry point that runs the module using `Module.run_from_registry()`. Handles both local and bundled import paths for the classifier.

- **src/models/classifier.py**: Core implementation of the Vision service
  - Implements the Viam `Vision` service interface using `EasyResource`
  - Requires an ML model dependency (`mlmodel_name` attribute)
  - Optionally depends on a camera (`camera_name` attribute)
  - Currently implements:
    - `capture_all_from_camera()` - returns images from camera
    - `get_properties()` - reports classifications_supported=True
  - Not yet implemented:
    - `get_classifications()`, `get_classifications_from_camera()`
    - `get_detections()`, `get_detections_from_camera()`
    - `get_object_point_clouds()`
    - `do_command()`

### Resource Configuration

The classifier is configured with attributes:
- `mlmodel_name` (required): Name of the ML model service to use
- `camera_name` (optional): Default camera to use when camera_name parameter is empty

### Dependency Management

The Viam SDK's dependency system is used extensively:
- Dependencies are declared in `validate_config()`
- Dependencies are resolved through the `dependencies` mapping in `new()` and `reconfigure()`
- Resources are retrieved using `ResourceName` lookups (e.g., `MLModel.get_resource_name()`)

## Python Version

This project requires Python 3.11+ (specified in pyproject.toml and .python-version).

## Deployment

The module is designed to run on multiple platforms:
- linux/amd64, linux/arm64
- darwin/arm64
- windows/amd64

The entry point is `dist/main` (the PyInstaller-built executable).

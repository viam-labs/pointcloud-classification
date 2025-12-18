import numpy as np

from src.utils.pointcloud import PointCloud, parse_pcd_bytes


def test_pointcloud_with_points_only():
    """Test creating PointCloud with only XYZ points"""
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    pcd = PointCloud(points=points)

    assert np.array_equal(pcd.points, points)
    assert pcd.colors is None
    assert pcd.normals is None
    assert not pcd.has_colors()
    assert not pcd.has_normals()


def test_pointcloud_with_colors():
    """Test creating PointCloud with XYZ and RGB"""
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    colors = np.array([[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]])

    pcd = PointCloud(points=points, colors=colors)

    assert np.array_equal(pcd.points, points)
    assert np.array_equal(pcd.colors, colors)
    assert pcd.normals is None
    assert pcd.has_colors()
    assert not pcd.has_normals()


def test_pointcloud_with_all_attributes():
    """Test creating PointCloud with XYZ, RGB, and normals"""
    points = np.array([[1.0, 2.0, 3.0]])
    colors = np.array([[0.5, 0.5, 0.5]])
    normals = np.array([[0.0, 0.0, 1.0]])

    pcd = PointCloud(points=points, colors=colors, normals=normals)

    assert np.array_equal(pcd.points, points)
    assert np.array_equal(pcd.colors, colors)
    assert np.array_equal(pcd.normals, normals)
    assert pcd.has_colors()
    assert pcd.has_normals()


def test_pointcloud_len():
    """Test that len() returns number of points"""
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    pcd = PointCloud(points=points)

    # This mimics open3d's len(pcd.points) pattern
    assert len(pcd.points) == 3


def test_parse_pcd_bytes_ascii_xyz():
    """Test parsing ASCII PCD format with XYZ only"""
    # Create a simple ASCII PCD file
    pcd_content = b"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH 3
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 3
DATA ascii
1.0 2.0 3.0
4.0 5.0 6.0
7.0 8.0 9.0
"""

    pcd = parse_pcd_bytes(pcd_content)

    assert isinstance(pcd, PointCloud)
    assert pcd.points.shape == (3, 3)
    assert np.allclose(pcd.points[0], [1.0, 2.0, 3.0])
    assert np.allclose(pcd.points[1], [4.0, 5.0, 6.0])
    assert np.allclose(pcd.points[2], [7.0, 8.0, 9.0])
    assert not pcd.has_colors()
    assert not pcd.has_normals()


def test_parse_pcd_bytes_ascii_xyzrgb():
    """Test parsing ASCII PCD format with XYZ and RGB"""
    pcd_content = b"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH 2
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 2
DATA ascii
1.0 2.0 3.0 4278190080
4.0 5.0 6.0 4294901760
"""

    pcd = parse_pcd_bytes(pcd_content)

    assert isinstance(pcd, PointCloud)
    assert pcd.points.shape == (2, 3)
    assert np.allclose(pcd.points[0], [1.0, 2.0, 3.0])
    assert pcd.has_colors()
    assert pcd.colors.shape == (2, 3)
    # RGB values should be in [0, 1] range
    assert pcd.colors.min() >= 0.0
    assert pcd.colors.max() <= 1.0


def test_parse_pcd_bytes_binary():
    """Test parsing binary PCD format with XYZ"""
    # Create a binary PCD header + data
    header = b"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH 2
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 2
DATA binary
"""
    # Binary data: 2 points with XYZ as float32 (little-endian)
    import struct

    point1 = struct.pack("<fff", 1.0, 2.0, 3.0)
    point2 = struct.pack("<fff", 4.0, 5.0, 6.0)
    pcd_content = header + point1 + point2

    pcd = parse_pcd_bytes(pcd_content)

    assert isinstance(pcd, PointCloud)
    assert pcd.points.shape == (2, 3)
    assert np.allclose(pcd.points[0], [1.0, 2.0, 3.0])
    assert np.allclose(pcd.points[1], [4.0, 5.0, 6.0])

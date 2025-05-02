import os
import sys
import pytest
import numpy as np
import trimesh
import pickle
from PIL import Image
from pathlib import Path

# python
from reproject.reproject_mesh.run_reprojection import (
    get_synced_meshes,
    render_mesh,
    project_meshes,
    get_camera_params,
    get_meshes
)

@pytest.fixture
def sample_data():
    return {
        'sync_file': 'path/to/sync.csv',
        'capture_dir': 'path/to/capture',
        'camera_calib': {
            'img_shape': (1080, 1920),
            'camera_matrix': np.eye(3),
            'xi': 1.0,
            'dist_coeffs': np.zeros(4)
        },
        'mesh': trimesh.creation.box(),
        'vertices_2d': np.random.rand(8, 1, 2),
        'image': Image.new('RGB', (1920, 1080))
    }

def test_render_mesh(sample_data, tmp_path):
    """Test mesh rendering function"""
    img = sample_data['image']
    output = render_mesh(
        img,
        'test.jpg',
        sample_data['mesh'],
        sample_data['vertices_2d'],
        output_dir=str(tmp_path)
    )
    assert output is not None
    assert isinstance(output, Image.Image)

def test_mesh_synchronization(sample_data):
    """Test mesh synchronization with images"""
    synced_files, synced_meshes = get_synced_meshes(
        sample_data['sync_file'],
        sample_data['capture_dir'],
        [sample_data['mesh']]
    )
    assert len(synced_files) == len(synced_meshes)

def test_camera_params(sample_data):
    """Test camera parameter extraction"""
    use_omni, cam_matrix, xi, dist = get_camera_params(sample_data['camera_calib'])
    assert isinstance(use_omni, bool)
    assert cam_matrix.shape == (3, 3)
    assert isinstance(xi, float)
    assert dist.shape == (4,)

def test_project_meshes(sample_data):
    """Test mesh projection"""
    vertices, meshes = project_meshes(
        'path/to/cam0',
        'path/to/cam1',
        [sample_data['mesh']],
        sample_data['sync_file'],
        sample_data['camera_calib'],
        render=False
    )
    assert len(vertices) == len(meshes)
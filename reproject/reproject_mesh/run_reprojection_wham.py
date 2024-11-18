#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reproject WHAM pose estimation outputs to omni camera using world coordinates.
Enhanced version with improved synchronization and debugging capabilities.
"""

import argparse
import os
import sys

sys.path.append('/home/sid/Projects/reprojection')
sys.path.append('/home/sid/Projects/humor/humor')

import pickle
import logging
from typing import Optional, Dict, List, Tuple
from pathlib import Path

import torch
import trimesh
import numpy as np
import joblib
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import required external functions (assuming these exist in your project)
from humor_inference.reproject_humor_sequence import (
    BodyModel,
    c2c,
    get_camera_params,
    transform_SMPL_sequence,
)

from reproject.reproject_mesh.reprojection_utils import (
    get_filepaths,
    get_calib_paths,
    get_kinect_list,
    get_transformation_matrix_opencv,
    get_transformation_matrix_matlab,
)

BM_PATH = "/home/sid/Projects/humor/body_models/smplh/male/model.npz"


class MeshSynchronizer:
    """Handles mesh synchronization between different camera    views."""
    
    def __init__(self, sync_file: str, capture_dir: str):
        self.sync_file = sync_file
        self.capture_dir = capture_dir
        self.sync_data = None
        self._load_sync_data()

    def _load_sync_data(self):
        """Load and validate synchronization data."""
        try:
            self.sync_data = pd.read_csv(self.sync_file, sep=";", header=0)
            logger.info(f"Loaded sync data with {len(self.sync_data)} entries")
        except Exception as e:
            logger.error(f"Failed to load sync file: {str(e)}")
            raise

    def get_synced_meshes(self, transformed_meshes: Dict[int, List[trimesh.Trimesh]], n: int = 0) -> Tuple[List, Dict]:
        """
        Get synchronized meshes and corresponding camera files.
        """
        capture_files = sorted([f for f in os.listdir(self.capture_dir) 
                              if f.endswith(('.png', '.jpg'))])
        
        # Create mapping of omni filenames to mesh indices
        synced_pairs = []
        transformed_meshes_dict = {}
        
        for _, row in self.sync_data.iterrows():
            capture_file = row[f"capture{n}"]
            omni_file = row["omni"]
            
            if capture_file in capture_files:
                mesh_idx = capture_files.index(capture_file)
                if mesh_idx in transformed_meshes:
                    transformed_meshes_dict[mesh_idx] = transformed_meshes[mesh_idx]
                    synced_pairs.append((omni_file, mesh_idx))
        
        # Sort by omni filename to maintain temporal order
        synced_pairs.sort(key=lambda x: x[0])
        synced_omni_files = [pair[0] for pair in synced_pairs]
        ordered_mesh_dict = {idx: transformed_meshes_dict[pair[1]] 
                           for idx, pair in enumerate(synced_pairs)}
        
        logger.info(f"Synchronized {len(synced_omni_files)} meshes with camera files")
        return synced_omni_files, ordered_mesh_dict


class MeshProcessor:
    """Handles mesh processing and transformation operations."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_wham_mesh_sequence(self, wham_output, betas, device):
        """
        Generates a dictionary of mesh sequences and frame IDs from WHAM output vertices and betas.

        Parameters:
        - wham_output (dict): The WHAM output containing multiple sequences.
        - betas (dict): Dictionary mapping sequence indices to shape coefficients.
        - device (torch.device): Device to perform computations on.

        Returns:
        - dict:
            - sequences (dict):
                - key (int): Sequence index.
                - value (dict):
                    - wham_meshes (List[trimesh.Trimesh]): List of mesh objects for each pose.
                    - frame_ids (List[int]): List of frame identifiers.
        """
        sequences = {}
        for seq_idx, seq_data in tqdm(wham_output.items()):
            verts = seq_data.get("verts")
            if verts is None:
                logger.error(f"'verts' not found for sequence {seq_idx}")
                continue

            betas_seq = betas.get(seq_idx)
            if betas_seq is None:
                logger.error(f"Betas not found for sequence {seq_idx}")
                continue

            T = betas_seq.shape[0]  # Number of poses for the current sequence
            num_betas = betas_seq.shape[1]  # Number of shape coefficients

            # Initialize the BodyModel with world parameters for the current sequence
            pred_bm = BodyModel(
                bm_path=BM_PATH,
                num_betas=num_betas,
                batch_size=T,
                model_type="smplh"  # Adjust based on your model type
            ).to(device)

            # Construct the meshes directly from WHAM vertices
            faces = pred_bm.bm.faces_tensor 
            faces = c2c(faces)  # Converts to NumPy

            try:
                wham_meshes = [
                    trimesh.Trimesh(
                        vertices=verts[i],
                        faces=faces,
                        process=False,
                    )
                    for i in range(len(verts))
                ]
            except IndexError as e:
                logger.error(f"Error creating meshes for sequence {seq_idx}: {e}")
                continue
            
            frame_ids = seq_data.get("frame_id", [])
            if not len(frame_ids):
                frame_ids = list(range(len(verts)))
            elif len(frame_ids) != len(verts):
                logger.error(f"The number of frame_ids does not match the number of meshes for sequence {seq_idx}.")
                frame_ids = list(range(len(verts)))  # Fallback to sequential IDs

            sequences[seq_idx] = {
                'wham_meshes': wham_meshes,
                'frame_ids': frame_ids
            }

        return sequences
    
    def sanitize_predictions(self, pred_res: Dict, T: int) -> Dict:
        """Sanitize WHAM predictions by handling NaN and infinite values."""
        for key, size in {"trans_world": 3, "betas": 10, "pose_world": 72}.items():
            if key not in pred_res:
                logger.warning(f"Key '{key}' not found in predictions")
                continue

            data = np.array(pred_res[key], dtype=np.float32)
            invalid_mask = ~np.isfinite(data)

            if np.any(invalid_mask):
                logger.warning(f"Found invalid values in {key}")
                if key == "betas":
                    data[invalid_mask] = 0.0
                else:
                    for t in range(T):
                        if np.all(invalid_mask[t]):
                            data[t] = 0.0
                        elif np.any(invalid_mask[t]):
                            valid_data = data[t][~invalid_mask[t]]
                            data[t][invalid_mask[t]] = np.nanmedian(valid_data) if valid_data.size > 0 else 0.0
                
                pred_res[key] = data

        return pred_res

    def load_meshes(self, results_folder: str) -> Dict:
        """Load WHAM mesh predictions from results folder."""
        wham_output_pkl_file = os.path.join(results_folder, "wham_output.pkl")
        if not os.path.isfile(wham_output_pkl_file):
            logger.error(f"Could not find {wham_output_pkl_file}!")
            return {}
    
        with open(wham_output_pkl_file, "rb") as file:
            wham_output = joblib.load(file)
    
        sanitized_output = {}
        betas_dict = {}
        logger.info("Sanitizing WHAM output")
        for seq_idx, seq_data in tqdm(wham_output.items()):
            sanitized_data = self.sanitize_predictions(
                seq_data,
                T=seq_data["verts"].shape[0]
            )
            sanitized_output[seq_idx] = sanitized_data
            verts, betas, _ = self.get_wham_parameters(sanitized_data)
            betas_dict[seq_idx] = betas

        logger.info(f"Collating WHAM meshes for {len(sanitized_output)} sequences")
        wham_meshes = self.get_wham_mesh_sequence(sanitized_output, betas_dict, self.device)
        aggregated_meshes = {}
        for seq_idx, seq_mesh_data in wham_meshes.items():
            for frame_id, mesh in zip(seq_mesh_data['frame_ids'], seq_mesh_data['wham_meshes']):
                if frame_id not in aggregated_meshes:
                    aggregated_meshes[frame_id] = []
                aggregated_meshes[frame_id].append(mesh)
    
        return aggregated_meshes

    def get_wham_parameters(self, wham_output):
        """
        Extracts vertices, betas, and device information from WHAM output.

        Parameters:
        - wham_output (dict): The WHAM output for a single sequence.

        Returns:
        - tuple:
            - verts (numpy.ndarray): Vertices from WHAM, shape (T, 6890, 3).
            - betas (numpy.ndarray): Shape coefficients, shape (T, 10).
            - device (torch.device): Computation device.
        """
        verts = wham_output["verts"]  # Shape: (T, 6890, 3)
        betas = wham_output["betas"]  # Shape: (T, 10)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return verts, betas, device

    def transform_meshes_old(self, pred_bodies: Dict[int, List[trimesh.Trimesh]], transform_matrix: np.ndarray) -> Dict[int, List[trimesh.Trimesh]]:
        """Transform meshes from Kinect to Omni camera frame."""
        transformed_meshes = {}
        logger.info(f"Transforming {len(pred_bodies)} meshes")
        for frame_id, meshes in tqdm(pred_bodies.items()):
            transformed_meshes[frame_id] = []
            for mesh in meshes:
                try:
                    transformed_mesh = mesh.copy()
                    transformed_mesh.apply_transform(transform_matrix)
                    transformed_meshes[frame_id].append(transformed_mesh)
                except Exception as e:
                    logger.error(f"Error transforming mesh: {str(e)}")
                    continue
        return transformed_meshes
    
    def transform_wham_to_humor(self, mesh, option='invert_yz'):
        """
        Transform the mesh from WHAM's coordinate system to HuMoR's coordinate system.
    
        Parameters:
        - mesh: trimesh.Trimesh instance of the mesh to transform.
        - option: str, specifies which transformation to apply. Possible values are:
            - 'invert_yz': Invert Y and Z axes.
            - 'invert_xz': Invert X and Z axes.
            - 'rotate_y_180': Rotate 180 degrees around Y-axis.
            - 'swap_yz': Swap Y and Z axes.
            - 'invert_z': Invert Z-axis.
    
        Returns:
        - mesh: transformed mesh.
        """
        import numpy as np
        import trimesh
    
        if option == 'invert_yz':
            # Invert Y and Z axes
            alignment_matrix = np.eye(4)
            alignment_matrix[1, 1] = -1  # Invert Y-axis
            alignment_matrix[2, 2] = -1  # Invert Z-axis
        elif option == 'invert_xz':
            # Invert X and Z axes
            alignment_matrix = np.eye(4)
            alignment_matrix[0, 0] = -1  # Invert X-axis
            alignment_matrix[2, 2] = -1  # Invert Z-axis
        elif option == 'rotate_y_180':
            # Rotate 180 degrees around the Y-axis
            alignment_matrix = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
        elif option == 'swap_yz':
            # Swap Y and Z axes
            alignment_matrix = np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ])
        elif option == 'invert_z':
            # Invert Z-axis
            alignment_matrix = np.eye(4)
            alignment_matrix[2, 2] = -1  # Invert Z-axis
        else:
            raise ValueError(f"Unknown option '{option}'")
    
        # Apply the transformation to the mesh
        mesh.apply_transform(alignment_matrix)
    
        return mesh
    def transform_meshes(self, pred_bodies, transform_matrix):
        transformed_meshes = {}
        logger.info(f"Transforming {len(pred_bodies)} meshes")
        for frame_id, meshes in tqdm(pred_bodies.items()):
            transformed_meshes[frame_id] = []
            for mesh in meshes:
                try:
                    transformed_mesh = mesh.copy()
                    # First align coordinate systems
                    # Options: 'invert_yz', 'invert_xz', 'rotate_y_180', 'swap_yz', 'invert_z'
                    transformed_mesh = self.transform_wham_to_humor(transformed_mesh, option='invert_z')
                    # Then apply the extrinsic transformation
                    transformed_mesh.apply_transform(transform_matrix)
                    transformed_meshes[frame_id].append(transformed_mesh)
                except Exception as e:
                    logger.error(f"Error transforming mesh: {str(e)}")
                    continue
        return transformed_meshes
    
    def analyze_transform(self, transform_matrix: np.ndarray):
        """Analyze transformation matrix properties"""
        print("Transform Matrix:")
        print(transform_matrix)
        print(f"Rotation matrix determinant: {np.linalg.det(transform_matrix[:3,:3])}")
        print(f"Translation magnitude: {np.linalg.norm(transform_matrix[:3,3])}")

    def debug_wham_structure(self, wham_output: Dict, seq_idx: int):
        """Debug WHAM output structure"""
        print(f"Sequence {seq_idx} stats:")
        print(f"Verts shape: {wham_output[seq_idx]['verts'].shape}")
        print(f"Trans_world shape: {wham_output[seq_idx]['trans_world'].shape}")
        print(f"Frame IDs: {wham_output[seq_idx]['frame_id'][:5]}...")
        print(f"Vertex range: {wham_output[seq_idx]['verts'].min():.3f} to {wham_output[seq_idx]['verts'].max():.3f}")

    def visualize_raw_mesh(self, mesh: trimesh.Trimesh, output_path: str):
        """Visualize raw mesh using matplotlib"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        vertices = mesh.vertices
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                  c='b', marker='.', s=1, alpha=0.5)
        
        for edge in mesh.edges:
            v1 = vertices[edge[0]]
            v2 = vertices[edge[1]]
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 
                    'k-', linewidth=0.1, alpha=0.3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Mesh Visualization')
        
        center = mesh.centroid
        radius = np.linalg.norm(vertices - center, axis=1).max()
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        
        plot_path = os.path.join(output_path, 'mesh_visualization.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        obj_path = os.path.join(output_path, 'mesh.obj')
        mesh.export(obj_path)
        
        logger.info(f"Saved mesh visualization to {plot_path}")
        logger.info(f"Saved mesh OBJ to {obj_path}")


class Renderer:
    """Handles mesh rendering and visualization."""
    
    def __init__(self, camera_calib: Dict):
        self.use_omni, self.camera_matrix, self.xi, self.dist_coeffs = get_camera_params(camera_calib)
        logger.info(f"Initialized renderer with camera calibration, using omni: {self.use_omni}")

    def project_vertices(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Project 3D vertices to 2D coordinates."""
        if self.use_omni:
            vertices_2d, _ = cv2.omnidir.projectPoints(
                np.expand_dims(mesh.vertices, axis=0),
                np.zeros(3),
                np.zeros(3),
                self.camera_matrix,
                self.xi,
                self.dist_coeffs,
            )
            return vertices_2d.swapaxes(0, 1)
        else:
            vertices_2d, _ = cv2.projectPoints(
                mesh.vertices,
                np.zeros(3),
                np.zeros(3),
                self.camera_matrix,
                self.dist_coeffs
            )
            return vertices_2d

    def render_mesh(self, img: Image.Image, mesh: trimesh.Trimesh, vertices_2d: np.ndarray) -> Image.Image:
        """Render mesh on image consistently."""
        # Remove the per-frame flipping
        draw = ImageDraw.Draw(img)
        try:
            for face in mesh.faces:
                face_vertices = vertices_2d[face]
                draw.polygon(
                    [tuple(p[0]) for p in face_vertices],
                    fill=None,
                    outline="gray",
                    width=1,
                )
        except Exception as e:
            logger.error(f"Error rendering mesh: {str(e)}")
            raise e
        return img
    
    def add_debug_markers(self, img: Image.Image, mesh: trimesh.Trimesh, color=(255,0,0)) -> Image.Image:
        """Add visual markers for key points using existing projection"""
        # Convert PIL Image to numpy array
        img_array = np.array(img)
        
        # For centroid
        centroid = mesh.centroid.reshape(1,3)
        centroid_2d = self.project_vertices(trimesh.Trimesh(vertices=centroid, faces=[]))
        # Fix: Extract x,y coordinates properly
        point = tuple(map(int, centroid_2d[0,0]))  # Get (x,y) from [[x,y]]
        cv2.circle(img_array, point, 5, color, -1)
        
        # For bounding box corners
        bbox_points = mesh.bounds.reshape(-1,3)
        bbox_2d = self.project_vertices(trimesh.Trimesh(vertices=bbox_points, faces=[]))
        for point_2d in bbox_2d:
            # Fix: Extract x,y coordinates properly
            point = tuple(map(int, point_2d[0]))  # Get (x,y) from [x,y]
            cv2.circle(img_array, point, 3, color, -1)
            
        return Image.fromarray(img_array)


    def debug_projection(self, mesh: trimesh.Trimesh, transform_matrix: np.ndarray):
        """Debug projection pipeline"""
        print(f"Original mesh centroid: {mesh.centroid}")
        
        transformed = mesh.copy()
        transformed.apply_transform(transform_matrix)
        print(f"Transformed centroid: {transformed.centroid}")
        
        test_points = transformed.vertices[[0, 100, 1000, -1]]
        print("Projected test points:")
        vertices_2d = self.project_vertices(
            trimesh.Trimesh(vertices=test_points, faces=[])
        )
        print(vertices_2d)


def main(args):
    # Initialize processors
    omni_calib_path = args.omni_intrinsics
    cam0_images_path, cam1_images_path, sync_file, output_path, results_folder = get_filepaths(args.root, args.n, args)
    kinect_matlab_jsonpath, omni_matlab_jsonpath, cam0_to_world_pth, world_to_cam1_pth = get_calib_paths(
        args.root, args.use_matlab, args.n)

    # cam0_to_world_pth = "/home/sid/Projects/reprojection/calibration/opencv_calibration/k0_rgb_cam_to_world.pkl"
    # world_to_cam1_pth = "/home/sid/Projects/reprojection/calibration/opencv_calibration/k0_omni_world_to_cam.pkl"

    mesh_sync = MeshSynchronizer(sync_file, cam0_images_path)
    mesh_processor = MeshProcessor()
    renderer = Renderer(pickle.load(open(args.omni_intrinsics, "rb")))
    
    # Load and process meshes
    logger.info("Loading WHAM meshes...")
    results_folder = args.results_folder
    pred_bodies = mesh_processor.load_meshes(results_folder)
    
    if not pred_bodies:
        raise ValueError("No valid meshes found in results folder")
        
    # Debug first mesh if requested
    if args.debug:
        first_frame = list(pred_bodies.keys())[0]
        first_mesh = pred_bodies[first_frame][0]
        mesh_processor.visualize_raw_mesh(first_mesh, args.output_dir)
    
    # Get and verify transformation matrix
    transform_matrix = (get_transformation_matrix_matlab if args.use_matlab 
                       else get_transformation_matrix_opencv)(cam0_to_world_pth, world_to_cam1_pth)
    
    if args.debug:
        mesh_processor.analyze_transform(transform_matrix)
    
    # Transform meshes
    transformed_meshes = mesh_processor.transform_meshes(pred_bodies, transform_matrix)
    
    # Synchronize meshes
    synced_files, synced_meshes = mesh_sync.get_synced_meshes(transformed_meshes, args.n)
    
    if not synced_files:
        raise ValueError("No synchronized frames found")

    # Render if requested
    if args.render:
        logger.info(f"Rendering {len(synced_files)} meshes to video")
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Setup video writer
        first_img_path = os.path.join(cam1_images_path, synced_files[0])
        img = Image.open(first_img_path)
        img = ImageOps.mirror(img)
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        height, width, layers = frame.shape
        
        video_path = Path(args.output_dir) / "output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(str(video_path), fourcc, 30, (width, height))
        
        # Process frames
        for frame_idx in tqdm(range(len(synced_files)), desc="Rendering frames"):
            img_path = synced_files[frame_idx]
            meshes = synced_meshes[frame_idx]
            
            # Load and process image
            img = Image.open(os.path.join(cam1_images_path, img_path))
            # Mirror image horizontally since extrinsics were built for mirrored images
            img = ImageOps.mirror(img)

            
            # Render each mesh in the frame
            for mesh in meshes:
                try:
                    # Project vertices
                    vertices_2d = renderer.project_vertices(mesh)
                    
                    # Debug projection if requested
                    if args.debug and frame_idx == 0:
                        renderer.debug_projection(mesh, transform_matrix)
                    
                    # Render mesh
                    img = renderer.render_mesh(img, mesh, vertices_2d)
                    
                    # Add debug markers if requested
                    if args.debug:
                        img = renderer.add_debug_markers(img, mesh)
                        
                except Exception as e:
                    logger.error(f"Error rendering mesh in frame {frame_idx}: {str(e)}")
                    continue
            
            # Convert to BGR for video writing
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            video.write(frame)
            
            # Save debug frame if requested
            if args.debug and frame_idx == 0:
                debug_path = Path(args.output_dir) / "debug_frame.png"
                cv2.imwrite(str(debug_path), frame)
        
        video.release()
        logger.info(f"Video saved to {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced WHAM mesh reprojection tool")
    parser.add_argument(
        "--root",
        type=str, 
        help="Root directory of the dataset", 
        default="/home/NAS-mountpoint/kinect-omni-ego/2023-02-09/at-unis/lab/a01"
    )
    parser.add_argument(
        "--omni_intrinsics", 
        type=str, 
        help="Path to the omni intrinsics file", 
        default="/home/sid/Projects/reprojection/calibration/intrinsics/omni_calib.pkl"
    )
    parser.add_argument(
        "--use-matlab",
        action="store_true",
        help="Use matrices from MATLAB instead of OpenCV",
    )
    parser.add_argument(
        "--partial-meshes",
        action="store_true"
    )   
    parser.add_argument(
        "--render",
        action="store_true",
    )
    parser.add_argument(
        "--n",
        type=int,
        help="Capture number, if only one is to be reprojected",
        default=0,
    )
    parser.add_argument(
        "--results-folder",
        type=str,
        help="Path to the WHAM results folder",
        default="/home/sid/Projects/WHAM/output/demo/2023-02-09_at-unis_lab_a01_capture0"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save rendered images and projections",
        default="/home/sid/Projects/WHAM/output/demo/dump",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debugging mode",
    )
    args = parser.parse_args()
    main(args)
#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.
"""
Load a HUMOR sequence fitted on one camera frame, and transform it to another, then render the mesh
sequence on the corresponding image sequence.
"""

import argparse
import os
import pickle
from typing import Dict, List, Optional
import cv2 as cv
import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw
from smplx import SMPL, SMPLH, SMPLX
from smplx.utils import Struct
from smplx.vertex_ids import vertex_ids
from tqdm import tqdm


# ============================== From the humor code base ===================================
class BodyModel(torch.nn.Module):
    """
    Wrapper around SMPLX body model class.
    """

    def __init__(
        self,
        bm_path,
        num_betas=10,
        batch_size=1,
        num_expressions=10,
        use_vtx_selector=False,
        model_type="smplh",
    ):
        super(BodyModel, self).__init__()
        """
        Creates the body model object at the given path.

        :param bm_path: path to the body model pkl file
        :param num_expressions: only for smplx
        :param model_type: one of [smpl, smplh, smplx]
        :param use_vtx_selector: if true, returns additional vertices as joints that correspond to OpenPose joints
        """
        self.use_vtx_selector = use_vtx_selector
        cur_vertex_ids = None
        if self.use_vtx_selector:
            cur_vertex_ids = vertex_ids[model_type]
        data_struct = None
        if ".npz" in bm_path:
            # smplx does not support .npz by default, so have to load in manually
            smpl_dict = np.load(bm_path, encoding="latin1")
            data_struct = Struct(**smpl_dict)
            # print(smpl_dict.files)
            if model_type == "smplh":
                data_struct.hands_componentsl = np.zeros((0))
                data_struct.hands_componentsr = np.zeros((0))
                data_struct.hands_meanl = np.zeros((15 * 3))
                data_struct.hands_meanr = np.zeros((15 * 3))
                V, D, B = data_struct.shapedirs.shape
                data_struct.shapedirs = np.concatenate(
                    [data_struct.shapedirs, np.zeros((V, D, SMPL.SHAPE_SPACE_DIM - B))],
                    axis=-1,
                )  # super hacky way to let smplh use 16-size beta
        kwargs = {
            "model_type": model_type,
            "data_struct": data_struct,
            "num_betas": num_betas,
            "batch_size": batch_size,
            "num_expression_coeffs": num_expressions,
            "vertex_ids": cur_vertex_ids,
            "use_pca": False,
            "flat_hand_mean": True,
        }
        assert model_type in ["smpl", "smplh", "smplx"]
        if model_type == "smpl":
            self.bm = SMPL(bm_path, **kwargs)
            self.num_joints = SMPL.NUM_JOINTS
        elif model_type == "smplh":
            self.bm = SMPLH(bm_path, **kwargs)
            self.num_joints = SMPLH.NUM_JOINTS
        elif model_type == "smplx":
            self.bm = SMPLX(bm_path, **kwargs)
            self.num_joints = SMPLX.NUM_JOINTS

        self.model_type = model_type

    def forward(
        self,
        root_orient=None,
        pose_body=None,
        pose_hand=None,
        pose_jaw=None,
        pose_eye=None,
        betas=None,
        trans=None,
        dmpls=None,
        expression=None,
        return_dict=False,
        **kwargs,
    ):
        """
        Note dmpls are not supported.
        """
        assert dmpls is None
        out_obj = self.bm(
            betas=betas,
            global_orient=root_orient,
            body_pose=pose_body,
            left_hand_pose=None
            if pose_hand is None
            else pose_hand[:, : (SMPLH.NUM_HAND_JOINTS * 3)],
            right_hand_pose=None
            if pose_hand is None
            else pose_hand[:, (SMPLH.NUM_HAND_JOINTS * 3) :],
            transl=trans,
            expression=expression,
            jaw_pose=pose_jaw,
            leye_pose=None if pose_eye is None else pose_eye[:, :3],
            reye_pose=None if pose_eye is None else pose_eye[:, 3:],
            return_full_pose=True,
            **kwargs,
        )

        out = {
            "v": out_obj.vertices,
            "f": self.bm.faces_tensor,
            "betas": out_obj.betas,
            "Jtr": out_obj.joints,
            "pose_body": out_obj.body_pose,
            "full_pose": out_obj.full_pose,
        }
        if self.model_type in ["smplh", "smplx"]:
            out["pose_hand"] = torch.cat(
                [out_obj.left_hand_pose, out_obj.right_hand_pose], dim=-1
            )
        if self.model_type == "smplx":
            out["pose_jaw"] = out_obj.jaw_pose
            out["pose_eye"] = pose_eye

        if not self.use_vtx_selector:
            # don't need extra joints
            out["Jtr"] = out["Jtr"][:, : self.num_joints + 1]  # add one for the root

        if not return_dict:
            out = Struct(**out)

        return out


SMPL_JOINTS = {
    "hips": 0,
    "leftUpLeg": 1,
    "rightUpLeg": 2,
    "spine": 3,
    "leftLeg": 4,
    "rightLeg": 5,
    "spine1": 6,
    "leftFoot": 7,
    "rightFoot": 8,
    "spine2": 9,
    "leftToeBase": 10,
    "rightToeBase": 11,
    "neck": 12,
    "leftShoulder": 13,
    "rightShoulder": 14,
    "head": 15,
    "leftArm": 16,
    "rightArm": 17,
    "leftForeArm": 18,
    "rightForeArm": 19,
    "leftHand": 20,
    "rightHand": 21,
}

SMPL_SIZES = {"trans": 3, "betas": 10, "pose_body": 63, "root_orient": 3}


def c2c(tensor):
    return tensor.detach().cpu().numpy()


def prep_res(np_res, device, T):
    """
    Load np result dict into dict of torch objects for use with SMPL body model.
    """
    betas = np_res["betas"]
    betas = torch.Tensor(betas).to(device)
    if len(betas.size()) == 1:
        num_betas = betas.size(0)
        betas = betas.reshape((1, num_betas)).expand((T, num_betas))
    else:
        num_betas = betas.size(1)
        assert betas.size(0) == T
    trans = np_res["trans"]
    trans = torch.Tensor(trans).to(device)
    root_orient = np_res["root_orient"]
    root_orient = torch.Tensor(root_orient).to(device)
    pose_body = np_res["pose_body"]
    pose_body = torch.Tensor(pose_body).to(device)

    res_dict = {
        "betas": betas,
        "trans": trans,
        "root_orient": root_orient,
        "pose_body": pose_body,
    }

    for k, v in np_res.items():
        if k not in ["betas", "trans", "root_orient", "pose_body"]:
            res_dict[k] = v
    return res_dict


def run_smpl(res_dict, body_model):
    smpl_body = body_model(
        pose_body=res_dict["pose_body"],
        pose_hand=None,
        betas=res_dict["betas"],
        root_orient=res_dict["root_orient"],
        trans=res_dict["trans"],
    )
    return smpl_body


# =================== My code starts here =============================


def transform_SMPL_sequence(
    body,
    transform: np.ndarray,
) -> List[trimesh.Trimesh]:
    """
    Body model:body model output from SMPL forward pass (where the sequence is the batch)
    - .v: vertices
    - .f: faces
    - .Jtr: joints
    """

    """ The meat of the function """
    faces = c2c(body.f)
    body_mesh_seq = [
        trimesh.Trimesh(
            vertices=c2c(body.v[i]),
            faces=faces,
            process=False,
        )
        for i in range(body.v.size(0))
    ]
    """ =============================== """
    print(f"[*] Processing sequence of {len(body_mesh_seq)} frames...")
    t_body_mesh_seq = []
    for body_mesh in tqdm(body_mesh_seq, total=len(body_mesh_seq)):
        # Apply transform
        body_mesh.apply_transform(transform)
        t_body_mesh_seq.append(body_mesh)
    return t_body_mesh_seq


def make_44(pose):
    return np.vstack((np.hstack((pose["R"], pose["t"])), [0, 0, 0, 1]))


J_BODY = len(SMPL_JOINTS) - 1  # no root


def export_timestamped_mesh_seq(
    images_dir: str,
    mesh_seq: List[trimesh.Trimesh],
):
    # Load all images in the directory
    timestamps = sorted(
        [
            fname.split(".")[0]
            for fname in os.listdir(images_dir)
            if fname.endswith(".png") or fname.endswith(".jpg")
        ]
    )
    assert len(timestamps) > 0, f"No images found in {images_dir}"
    if len(timestamps) > len(mesh_seq):
        before_len = len(timestamps)
        timestamps = timestamps[: len(mesh_seq)]
        after_len = len(timestamps)
        print(f"[!] Warning: more images than meshes, truncating images to match. ({before_len} -> {after_len})")
    elif len(timestamps) < len(mesh_seq):
        before_len = len(mesh_seq)
        mesh_seq = mesh_seq[: len(timestamps)]
        after_len = len(mesh_seq)
        print(f"[!] Warning: more meshes than images, truncating meshes to match. ({before_len} -> {after_len})")
    assert len(timestamps) == len(mesh_seq)
    return {int(ts): mesh for ts, mesh in zip(timestamps, mesh_seq)}


def render_on_images(
    images_dir: str,
    mesh_seq: List[trimesh.Trimesh],
    camera_calib: dict,
    output_dir: str,
):
    """
    Render the mesh sequence on the images in the directory.
    """
    # Load the camera intrinsics and distortion coefficients from the pickle file
    camera_matrix = camera_calib["intrinsics"]
    dist_coeffs = camera_calib["distortion"]
    xi = camera_calib.get("xi")
    use_omni = xi is not None
    xi = xi.item() if isinstance(xi, np.ndarray) else xi

    # Load all images in the directory
    images = []
    for filename in os.listdir(images_dir):
        # If the file type is an image, load it
        if filename.endswith(".png") or filename.endswith(".jpg"):
            images.append(os.path.join(images_dir, filename))
    images.sort()
    # assert len(mesh_seq) == len(images), "Number of images and meshes must be the same!"
    if len(images) > len(mesh_seq):
        images = images[: len(mesh_seq)]
        print("Warning: more images than meshes, truncating images to match.")
    else:
        mesh_seq = mesh_seq[: len(images)]
        print("Warning: more meshes than images, truncating meshes to match.")
    # For each image in the folder, project the mesh on it using opencv's projectPoints.
    # Then, render the mesh on the image using PIL.
    for i, (mesh, img_path) in tqdm(
        enumerate(zip(mesh_seq, images)), total=len(images)
    ):
        img = Image.open(img_path)
        assert (img.size[1], img.size[0]) == camera_calib[
            "img_shape"
        ], "Image shape must match the camera calibration!"
        # Project the vertices on the image
        # We've already transformed the vertices to the camera frame, so we can just
        # project them on the image without any transformation.
        if use_omni:
            vertices_2d, _ = cv.omnidir.projectPoints(
                np.expand_dims(mesh.vertices, axis=0),
                np.zeros(3),
                np.zeros(3),
                camera_matrix,
                xi,
                dist_coeffs,
            )
            vertices_2d = np.swapaxes(vertices_2d, 0, 1)
        else:
            vertices_2d, _ = cv.projectPoints(
                mesh.vertices, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs
            )
        # Draw the mesh on the image
        draw = ImageDraw.Draw(img)
        for face in mesh.faces:
            face_vertices = vertices_2d[face]
            draw.polygon(
                [tuple(p[0]) for p in face_vertices],
                fill=None,
                outline="gray",
                width=1,
            )
        # Save the image
        img.save(os.path.join(output_dir, f"{i:08d}.png"))
    print("[*] Done!")


def sanitize_preds(pred_res, T):
    """
    Sanitize the predictions from the SMPL model.
    """
    # check if have any nans valid
    for smpk in SMPL_SIZES.keys():
        # TODO: We may want to do something different here! Like skip the frames or whatever.
        cur_valid = (
            torch.sum(
                torch.logical_not(torch.isfinite(torch.Tensor(pred_res[smpk])))
            ).item()
            == 0
        )
        if not cur_valid:
            print(f"Found NaNs in prediction for {smpk}, filling with zeros...")
            # print(pred_res[smpk].shape)
            if smpk == "betas":
                pred_res[smpk] = np.zeros((pred_res[smpk].shape[0]), dtype=np.float32)
            else:
                pred_res[smpk] = np.zeros(
                    (T, pred_res[smpk].shape[1]), dtype=np.float32
                )


def main(
    cam0_to_world_pth: str,
    world_to_cam1_pth: str,
    humor_output_path: str,
    images_path: str,
    output_path: Optional[str] = None,
    cam1_calib_pth: Optional[str] = None,
) -> Optional[Dict[int, trimesh.Trimesh]]:
    with open(cam0_to_world_pth, "rb") as f:
        cam0_to_world = make_44(pickle.load(f))
    with open(world_to_cam1_pth, "rb") as f:
        world_to_cam1 = make_44(pickle.load(f))

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    results_dir = os.path.join(humor_output_path, "results_out", "final_results")
    res_file = os.path.join(results_dir, "stage3_results.npz")
    if not os.path.isfile(res_file):
        return None
    pred_res = np.load(res_file)
    T = pred_res["trans"].shape[0]
    sanitize_preds(pred_res, T)
    pred_res = prep_res(pred_res, device, T)
    num_pred_betas = pred_res["betas"].size(1)

    # create body models for each
    meta_path = os.path.join(results_dir, "meta.txt")
    if not os.path.exists(meta_path):
        raise Exception(f"Could not find {meta_path}!")
    optim_bm_path = None
    with open(meta_path, "r") as f:
        optim_bm_str = f.readline().strip()
        optim_bm_path = optim_bm_str.split(" ")[1]

    # humor model
    pred_bm = BodyModel(
        bm_path=optim_bm_path, num_betas=num_pred_betas, batch_size=T
    ).to(device)

    # run through SMPL
    pred_body = run_smpl(pred_res, pred_bm)
    print("[*] Loaded the sequence of SMPL models!")

    with open(cam0_to_world_pth, "rb") as f:
        cam0_to_world = make_44(pickle.load(f))
    with open(world_to_cam1_pth, "rb") as f:
        world_to_cam1 = make_44(pickle.load(f))

    transform = world_to_cam1 @ cam0_to_world
    transform[:3, 3] = transform[:3, 3] / 1000.0

    print("[*] Applying the transform to the SMPL models sequence...")
    transformed_meshes = transform_SMPL_sequence(pred_body, transform)

    if cam1_calib_pth is None:
        # Return a dictionary of the transformed meshes where the keys are the matching image names (timestamps)
        return export_timestamped_mesh_seq(images_path, transformed_meshes)
    with open(cam1_calib_pth, "rb") as f:
        cam1_calib = pickle.load(f)
    output_path = output_path if output_path is not None else "projected_output_viz"
    os.makedirs(output_path, exist_ok=True)
    render_on_images(images_path, transformed_meshes, cam1_calib, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "humor_results_dir",
        type=str,
        help="Path to the directory containing the humor results.",
    )
    parser.add_argument(
        "images_dir",
        type=str,
        help="Path to the images directory from fitting.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        default="transformed_output",
        help="Path to save visualizations to. If not set, won't visualize.",
    )
    parser.add_argument(
        "--cam0-to-world",
        type=str,
        dest="cam0_to_world",
        help="Pickle file of [R|t] matrix from camera 0 to world coordinates.",
        required=True,
    )
    parser.add_argument(
        "--world-to-cam1",
        type=str,
        dest="world_to_cam1",
        help="Pickle file of [R|t] matrix from world to camera 1 coordinates.",
        required=True,
    )
    parser.add_argument(
        "--cam1-calib",
        type=str,
        dest="cam1_calib",
        help="Pickle file of camera 1 calibration (intrinsics, distortion, etc.). If not set, won't visualize",
        required=False,
    )
    args = parser.parse_args()
    main(
        args.cam0_to_world,
        args.world_to_cam1,
        args.humor_results_dir,
        args.images_dir,
        output_path=args.output_dir,
        cam1_calib_pth=args.cam1_calib,
    )

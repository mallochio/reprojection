#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   train.py
@Time    :   2024/03/08 18:29:35
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
<<<<<<< HEAD
@Desc    :   Script to train the model using torch (TODO - This is a stub, needs to be` filled in)
'''
import os
import json
import pytorch_lightning as pl
from pytorch_lightning import Trainer
=======
@Desc    :   Training code for new pose estimation models
"""
import os
import json
>>>>>>> e9d7114046457c0f7e015d949368ccd983bff08c
import torch
import numpy as np
from PIL import Image
import trimesh
import argparse
<<<<<<< HEAD
=======
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from resnet_transfomer_concat import PoseEstimationModule
import torchvision.transforms as T
>>>>>>> e9d7114046457c0f7e015d949368ccd983bff08c

torch.set_float32_matmul_precision("high")  # or 'medium'

<<<<<<< HEAD
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Open as RGB
    image = apply_image_transformations(image)  # Preprocessing function
    return image 

def apply_image_transformations(image):
    transform = T.Compose([
        T.Resize(256),  # Resize image
        T.CenterCrop(224),  # Center crop for consistency
        T.ToTensor(),  # Convert to PyTorch tensor
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    return transform(image)

def calculate_pose_smoothness(output_poses):
    # Simple temporal smoothness - difference between consecutive frames
    smoothness_loss = torch.mean(torch.norm(output_poses[:-1] - output_poses[1:], dim=(1, 2))) 
    return smoothness_loss

class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, base_dir):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.base_dir = base_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        image_folder = os.path.join(self.base_dir, item['omni'])
        image_files = os.listdir(image_folder)
        image_path = os.path.join(image_folder, image_files[0])  # Assuming the first image in the folder
        image = load_image(image_path)  

        pose_folder = os.path.join(self.base_dir, item['pose'])
        pose_files = os.listdir(pose_folder)
        pose_path = os.path.join(pose_folder, pose_files[0])  # Assuming the first pose file in the folder
        poses = np.load(pose_path)

        betas = torch.from_numpy(poses['betas']).float()  
        canonical_pose = torch.from_numpy(poses['pose']).float() 
        camera_pose = torch.from_numpy(poses['camera']).float()

        mesh_folder = os.path.join(self.base_dir, item['mesh'])
        mesh_files = os.listdir(mesh_folder)
        mesh_path = os.path.join(mesh_folder, mesh_files[0])  # Assuming the first mesh file in the folder
        target_mesh = trimesh.load(mesh_path)
        target_mesh = smplx.from_trimesh(target_mesh, model_type='smplh', gender='neutral')   

        return image, betas, camera_pose, canonical_pose, target_mesh  

class PoseEstimationModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, smpl_model_path='/raid/aml4aha/sid/humor/body_models/smplh/neutral/model.npz'):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True) 
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  
        self.smpl = smplx.create(smpl_model_path, model_type='smplh', gender='neutral') 
        num_pose_parameters = 63 # 3 parameters per body joint x 21 body joints For the body pose (make it 63+3 to add global orientation)

        self.mesh_regressor = nn.Sequential(
             nn.Linear(512, 256),
             nn.ReLU(),
             nn.Linear(256, self.smpl.num_betas) 
=======

class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, base_dir):
        with open(json_file, "r") as f:
            self.data = json.load(f)
        self.base_dir = base_dir
        self.all_files = self.prepare_dataset()

    def __len__(self):
        return len(self.all_files)

    def prepare_dataset(self):
        all_files = []
        for item in self.data:
            image_folder = os.path.join(self.base_dir, item["omni"])
            pose_folder = os.path.join(self.base_dir, item["pose"])
            mesh_folder = os.path.join(self.base_dir, item["mesh"])

            image_files = sorted(os.listdir(image_folder))
            pose_files = sorted(os.listdir(pose_folder))
            mesh_files = sorted(os.listdir(mesh_folder))

            # pair each image file with corresponding pose and mesh file
            all_files.extend(
                {
                    "image": os.path.join(image_folder, img_file),
                    "pose": os.path.join(pose_folder, pose_file),
                    "mesh": os.path.join(mesh_folder, mesh_file),
                }
                for img_file, pose_file, mesh_file in zip(
                    image_files, pose_files, mesh_files
                )
            )
        return all_files

    def __getitem__(self, index):
        files = self.all_files[index]
        image = self.load_image(files["image"])
        poses = torch.from_numpy(np.load(files["pose"])).float()
        target_mesh = trimesh.load(files["mesh"], process=False)
        target_vertices = torch.tensor(target_mesh.vertices).float()
        return image, poses, target_vertices

    def load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")  # Open as RGB
        image = self.apply_image_transformations(image)  # Preprocessing function
        return image

    def apply_image_transformations(self, image):
        transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),  # Center crop for consistency
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet normalization
            ]
>>>>>>> e9d7114046457c0f7e015d949368ccd983bff08c
        )
        return transform(image)

    def custom_collate_fn(self, batch):
        images = [item for sublist in [x[0] for x in batch] for item in sublist]
        projected_vertices = [
            item for sublist in [x[1] for x in batch] for item in sublist
        ]
        target_vertices = [
            item for sublist in [x[2] for x in batch] for item in sublist
        ]

        # Pad the sequences with zeros so they have the same length and can be batched
        images = pad_sequence(images, batch_first=True, padding_value=0)
        projected_vertices = pad_sequence(
            projected_vertices, batch_first=True, padding_value=0
        )
<<<<<<< HEAD
        self.learning_rate = learning_rate

    def forward(self, images, betas, camera_pose, canonical_pose):
        features = self.backbone(images)
        betas = self.mesh_regressor(features)

        output_mesh = self.smpl(betas=betas, 
                                global_orient=camera_pose[:, :3], 
                                pose2rot=False)  

        combined_features = torch.cat([features, betas, camera_pose, canonical_pose], dim=1) 
        output_poses = self.output_head(combined_features)
        return output_poses, output_mesh

    def training_step(self, batch, batch_idx):
        images, betas, camera_poses, canonical_poses, target_meshes = batch
        output_poses, output_mesh = self(images, betas, camera_poses, canonical_poses)
        loss = calculate_loss(output_poses, canonical_poses, output_mesh, target_meshes)  # Your loss function
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
=======
        target_vertices = pad_sequence(
            target_vertices, batch_first=True, padding_value=0
        )
        return images, projected_vertices, target_vertices
>>>>>>> e9d7114046457c0f7e015d949368ccd983bff08c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-file", default="/home/sid/Projects/NAS-mountpoint/train.json", type=str
    )
    parser.add_argument(
        "--base-dir", default="/home/sid/Projects/NAS-mountpoint", type=str
    )
    args = parser.parse_args()
    dataset = PoseDataset(args.json_file, args.base_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=20,
        # pin_memory=True,
        # prefetch_factor=2,
    )
    SMPL_MODEL_PATH = (
        "/home/sid/Projects/WHAM/dataset/body_models/smpl/SMPL_NEUTRAL.pkl"
    )
    model = PoseEstimationModule(smpl_model_path=SMPL_MODEL_PATH, learning_rate=0.1)
    wandb_logger = WandbLogger(
        log_model="all", project="Hades-krypton", entity="pyre", save_dir="/tmp/wandb"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",  # Metric to monitor
        mode="min",  # Condition: save on lower values
        save_top_k=1,  # Save only the single best model
        save_last=False,  # Optionally save the last model
    )
    trainer = Trainer(
        max_epochs=100,
        logger=wandb_logger,
        default_root_dir="/tmp/lightning_logs",
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, dataloader)
    torch.cuda.empty_cache()

    
if __name__ == "__main__":
<<<<<<< HEAD
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-file", default=None, type=str, required=True)
    parser.add_argument("--base-dir", default=None, type=str, required=True)
    args = parser.parse_args()

    JSON_FILE = args.json_file
    BASE_DIR = args.base_dir

    dataset = PoseDataset(JSON_FILE, BASE_DIR)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    model = PoseEstimationModule()
    trainer = Trainer(max_epochs=10, gpus=1)  
    trainer.fit(model, dataloader)
=======
    main()
>>>>>>> e9d7114046457c0f7e015d949368ccd983bff08c

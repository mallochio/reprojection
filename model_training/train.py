#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2024/02/09 19:28:47
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Script to train the model using torch (TODO - This is a stub, needs to be filled in)
'''

import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import smplx
import trimesh


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
    def __init__(self, image_dir, pose_dir, mesh_dir):
        self.image_dir = image_dir
        self.pose_dir = pose_dir
        self.mesh_dir = mesh_dir
        self.image_paths = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_paths[index])
        image = load_image(image_path)  

        pose_path = os.path.join(self.pose_dir, self.image_paths[index]).replace('.jpg', '.npy')  
        poses = np.load(pose_path)

        betas = torch.from_numpy(poses['betas']).float()  
        canonical_pose = torch.from_numpy(poses['pose']).float() 
        camera_pose = torch.from_numpy(poses['camera']).float()

        target_mesh_path = os.path.join(self.mesh_dir, self.image_paths[index]).replace('.jpg', '.obj')  
        target_mesh = trimesh.load(target_mesh_path)
        target_mesh = smplx.from_trimesh(target_mesh, model_type='smplh', gender='neutral')   

        return image, betas, camera_pose, canonical_pose, target_mesh  

class PoseEstimationModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, smpl_model_path='path/to/smplh/model.pkl'):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True) 
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  
        self.smpl = smplx.create(smpl_model_path, model_type='smplh', gender='neutral') 
        num_pose_parameters = 63 # 3 parameters per body joint x 21 body joints For the body pose (make it 63+3 to add global orientation)

        self.mesh_regressor = nn.Sequential(
             nn.Linear(512, 256),
             nn.ReLU(),
             nn.Linear(256, self.smpl.num_betas) 
        )

        self.output_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Linear(256, num_pose_parameters)  # Replace num_pose_parameters
        )
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
        images, camera_poses, canonical_poses, target_poses = batch
        output_poses = self(images, camera_poses, canonical_poses)
        loss = calculate_loss(output_poses, target_poses)  # Your loss function
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def calculate_loss(output_poses, target_poses, output_mesh, target_mesh):
    body_pose_loss = nn.L1Loss(reduction='mean')(output_poses[:, 3:66], target_poses[:, 3:66])
    body_shape_loss = nn.MSELoss(reduction='mean')(output_poses[:, 66:4328], target_poses[:, 66:4328])
    mesh_loss = nn.L1Loss(reduction='mean')(output_mesh.vertices, target_mesh.vertices)
    return body_pose_loss + body_shape_loss + mesh_loss


if __name__ == "__main__":
    DATASET_PATH = "path/to/dataset"
    POSE_PATH = "path/to/pose"
    MESH_PATH = "path/to/meshes" 

    dataset = PoseDataset(DATASET_PATH, POSE_PATH, MESH_PATH)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    model = PoseEstimationModule()
    trainer = Trainer(max_epochs=10, gpus=1)  
    trainer.fit(model, dataloader)
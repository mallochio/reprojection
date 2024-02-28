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
from pytorch_lightning.fabric import Fabric
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from torchvision import models
from PIL import Image


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


def calculate_loss(output_poses, target_poses):
    # Assuming your SMPL-H parameter order is standard
    body_pose_loss = nn.L1Loss(reduction='mean')(output_poses[:, :72], target_poses[:, :72])
    body_shape_loss = nn.MSELoss(reduction='mean')(output_poses[:, 72:82], target_poses[:, 72:82])
    left_hand_loss = nn.L1Loss(reduction='mean')(output_poses[:, 82:117], target_poses[:, 82:117])
    right_hand_loss = nn.L1Loss(reduction='mean')(output_poses[:, 117:152], target_poses[:, 117:152])
    face_loss = nn.L1Loss(reduction='mean')(output_poses[:, 152:], target_poses[:, 152:]) 

    total_loss = body_pose_loss + body_shape_loss + left_hand_loss + right_hand_loss + face_loss 
    return total_loss


class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, pose_paths):
        self.image_paths = image_paths
        self.pose_paths = pose_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = load_image(image_path)  

        poses = np.load(self.pose_paths[index])
        camera_pose = poses['camera']
        canonical_pose = poses['canonical']
        return image, camera_pose, canonical_pose


class PoseEstimationModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)  # Or any ResNet variant

        # Remove the last fully connected layer of ResNet
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1]) 

        self.output_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),  # Adjust based on ResNet's output features
            nn.ReLU(),
            nn.Linear(256, num_pose_parameters)  # Output SMPL parameters
        )
        self.learning_rate = learning_rate

    def forward(self, images, camera_pose, canonical_pose):
        features = self.backbone(images)
        # Optionally concatenate features with pose embeddings
        combined_features = torch.cat([features, camera_pose, canonical_pose], dim=1) 
        output_poses = self.output_head(combined_features)
        return output_poses


    def training_step(self, batch, batch_idx):
        images, camera_poses, canonical_poses, target_poses = batch
        output_poses = self(images, camera_poses, canonical_poses)
        loss = calculate_loss(output_poses, target_poses)  # Your loss function
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    # Example dataset setup
    DATASET_PATH = "path/to/dataset"
    POSE_PATH = "path/to/pose"

    image_paths = os.listdir(DATASET_PATH)
    pose_path = POSE_PATH
    dataset = PoseDataset(image_paths, pose_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    model = PoseEstimationModule()
    fabric = Fabric(accelerator="gpu", devices=2)
    trainer = pl.Trainer(fabric=fabric, max_epochs=10, ...)  # Add other trainer args

    trainer.fit(model, dataloader)

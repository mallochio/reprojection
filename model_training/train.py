#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2024/02/09 19:28:47
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Script to train the model using torch (TODO - This is a stub, needs to be` filled in)
'''
import os
import json
import lightning as pl
from lightning.pytorch import Trainer
import torch
import torch.nn as nn
import numpy as np
from transformers import ViTModel
import torchvision.transforms as T
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import smplx
import trimesh
import argparse
from lightning.pytorch.loggers import WandbLogger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


torch.set_float32_matmul_precision('high')  # or 'medium'

SMPL_MODEL_PATH = '/home/sid/Projects/WHAM/dataset/body_models/smpl/SMPL_NEUTRAL.pkl'

def custom_collate_fn(batch):
    images = [item for sublist in [x[0] for x in batch] for item in sublist]
    projected_vertices = [item for sublist in [x[1] for x in batch] for item in sublist]
    target_vertices = [item for sublist in [x[2] for x in batch] for item in sublist]

    # Pad the sequences with zeros so they have the same length and can be batched
    images = pad_sequence(images, batch_first=True, padding_value=0)
    projected_vertices = pad_sequence(projected_vertices, batch_first=True, padding_value=0)
    target_vertices = pad_sequence(target_vertices, batch_first=True, padding_value=0)
    return images, projected_vertices, target_vertices


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
        self.all_files = self.prepare_dataset()

    def prepare_dataset(self):
        all_files = []
        for item in self.data:
            image_folder = os.path.join(self.base_dir, item['omni'])
            pose_folder = os.path.join(self.base_dir, item['pose'])
            mesh_folder = os.path.join(self.base_dir, item['mesh'])

            image_files = sorted(os.listdir(image_folder))
            pose_files = sorted(os.listdir(pose_folder))
            mesh_files = sorted(os.listdir(mesh_folder))

            # pair each image file with corresponding pose and mesh file
            all_files.extend(
                {
                    'image': os.path.join(image_folder, img_file),
                    'pose': os.path.join(pose_folder, pose_file),
                    'mesh': os.path.join(mesh_folder, mesh_file),
                }
                for img_file, pose_file, mesh_file in zip(
                    image_files, pose_files, mesh_files
                )
            )
        return all_files

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        files = self.all_files[index]
        image = load_image(files['image'])  # Load a single image
        poses = torch.from_numpy(np.load(files['pose'])).float()  # Load a single pose
        target_mesh = trimesh.load(files['mesh'], process=False)  # Load a single mesh
        target_vertices = torch.tensor(target_mesh.vertices).float()  # Convert mesh to tensor
        return image, poses, target_vertices


class PoseEstimationModule_old(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, smpl_model_path=SMPL_MODEL_PATH):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True) 
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  
        self.smpl = smplx.create(model_path=smpl_model_path, model_type='smplh', gender='neutral') 
        num_pose_parameters = 63 # 3 parameters per body joint x 21 body joints For the body pose (make it 63+3 to add global orientation)

        self.pose_regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 69)  # Predict 69 pose parameters for SMPL-H model
        )

        self.output_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Linear(256, num_pose_parameters)  # Replace num_pose_parameters
        )
        self.learning_rate = learning_rate

    def forward(self, images):
        features = self.backbone(images)
        features = features.view(features.size(0), -1)  # Reshape features to (batch_size, 512)
        
        global_orient = torch.zeros((features.size(0), 3), device=features.device)  # Initialize global orientation to zeros
        body_pose = self.pose_regressor(features)
        output_mesh = self.smpl(global_orient=global_orient, body_pose=body_pose)
        output_vertices = output_mesh.vertices

        return output_vertices.view(output_vertices.size(0), -1, 3)  # Reshape output vertices to (batch_size, 6890, 3)

    def training_step(self, batch, batch_idx):
        images, projected_vertices, target_vertices = batch
        output_vertices = self(images)
        loss = self.calculate_loss(output_vertices, projected_vertices, target_vertices)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def calculate_loss(self, output_vertices, projected_vertices, target_vertices):
        # Ensure the projected vertices tensor has the same number of dimensions as the output
        if projected_vertices.dim() == 4 and projected_vertices.size(2) == 1:
            projected_vertices = projected_vertices.squeeze(2)

        # Now calculate vertex loss and projection loss
        vertex_loss = nn.MSELoss(reduction='mean')(output_vertices, target_vertices)
        projection_loss = nn.MSELoss(reduction='mean')(output_vertices[:, :, :2], projected_vertices)
        # log vertex loss and projection loss
        self.log("vertex_loss", vertex_loss)
        self.log("projection_loss", projection_loss)

        return vertex_loss + projection_loss


class PoseEstimationModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, smpl_model_path=SMPL_MODEL_PATH, vit_name='google/vit-base-patch16-224'):
        super().__init__()
        
        # Load pre-trained ResNet18 model with the new weights parameter
        resnet_weights = ResNet18_Weights.IMAGENET1K_V1
        self.resnet = resnet18(weights=resnet_weights)
        self.resnet_out_features = self.resnet.fc.in_features  # Save the number of output features before altering the model
        
        # Remove the fully connected layer from ResNet
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Load a pre-trained Vision Transformer
        self.vit = ViTModel.from_pretrained(vit_name)
        self.vit_out_features = self.vit.config.hidden_size
        
        # Define the output head with the combined features
        self.output_head = nn.Sequential(
            nn.Linear(self.resnet_out_features + self.vit_out_features, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 72)  # 72 pose parameters for SMPL
        )
        self.learning_rate = learning_rate

        # Initialize the SMPL model
        self.smpl = smplx.create(smpl_model_path, model_type='smpl')

        # Projection layer to map ResNet features to the desired dimension
        self.resnet_projection = nn.Linear(self.resnet_out_features, self.vit.config.hidden_size * (self.vit.config.image_size // self.vit.config.patch_size) ** 2)

    def forward(self, images):
        batch_size = images.size(0)

        # Extract features using the ResNet
        resnet_features = self.resnet(images)
        resnet_features = resnet_features.view(batch_size, -1)

        # Create dummy pixel values for ViT (or use actual image patches)
        dummy_pixel_values = torch.zeros(batch_size, 3, self.vit.config.image_size, self.vit.config.image_size, device=images.device)
        
        # Extract features using the ViT
        vit_outputs = self.vit(pixel_values=dummy_pixel_values)
        vit_features = vit_outputs.last_hidden_state[:, 0, :]
        
        # Combine features from ResNet and ViT
        combined_features = torch.cat((resnet_features, vit_features), dim=1)

        # Predict pose parameters
        pose_parameters = self.output_head(combined_features)
        
        # Split the pose parameters into global orientation and body pose
        global_orient = pose_parameters[:, :3]
        body_pose = pose_parameters[:, 3:]
        
        # Get the output mesh from the SMPL model
        output_mesh = self.smpl(global_orient=global_orient, body_pose=body_pose)
        output_vertices = output_mesh.vertices

        return output_vertices.view(batch_size, -1, 3)

    # Define the training step
    def training_step(self, batch, batch_idx):
        images, projected_vertices, target_vertices = batch
        output_vertices = self(images)
        loss = self.calculate_loss(output_vertices, projected_vertices, target_vertices)
        self.log("train_loss", loss)
        return loss

    # Define the optimizer
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


    def calculate_loss(self, output_vertices, projected_vertices, target_vertices):
        # Ensure the projected vertices tensor has the same number of dimensions as the output
        if projected_vertices.dim() == 4 and projected_vertices.size(2) == 1:
            projected_vertices = projected_vertices.squeeze(2)

        # Now calculate vertex loss and projection loss
        vertex_loss = nn.MSELoss(reduction='mean')(output_vertices, target_vertices)
        projection_loss = nn.MSELoss(reduction='mean')(output_vertices[:, :, :2], projected_vertices)

        return vertex_loss + projection_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-file", default="/home/sid/Projects/NAS-mountpoint/train.json", type=str)
    parser.add_argument("--base-dir", default="/home/sid/Projects/NAS-mountpoint", type=str)
    args = parser.parse_args()

    dataset = PoseDataset(args.json_file, args.base_dir)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=20, pin_memory=True)

    model = PoseEstimationModule()
    wandb_logger = WandbLogger(log_model="all", project="Hades-krypton", entity="pyre")
    trainer = Trainer(
        max_epochs=100,
        logger=wandb_logger,
    )
    trainer.fit(model, dataloader)
    torch.cuda.empty_cache()

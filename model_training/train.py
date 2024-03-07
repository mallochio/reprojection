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
import pytorch_lightning as pl
from pytorch_lightning import Trainer
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

SMPL_MODEL_PATH = '/home/sid/Projects/WHAM/dataset/body_models/smpl/SMPL_NEUTRAL.pkl'

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

        projected_vertices = torch.from_numpy(poses).float()  # Convert poses to tensor

        mesh_folder = os.path.join(self.base_dir, item['mesh'])
        mesh_files = os.listdir(mesh_folder)
        mesh_path = os.path.join(mesh_folder, mesh_files[0])  # Assuming the first mesh file in the folder
        target_mesh = trimesh.load(mesh_path, process=False)

        target_vertices = torch.tensor(target_mesh.vertices).float()

        return image, projected_vertices, target_vertices

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

    def forward(self, images):
        batch_size = images.size(0)
        # Extract features using the ResNet
        resnet_features = self.resnet(images).view(batch_size, -1)
        
        # Prepare "fake tokens" for ViT; ViT expects a sequence, so we need to create a sequence of length 1
        # fake_tokens = resnet_features.unsqueeze(1)
        vit_tokens = resnet_features.view(batch_size, 1, self.resnet_out_features)
        
        # Extract features using the ViT
        # vit_outputs = self.vit(pixel_values=fake_tokens)
        vit_outputs = self.vit(pixel_values=vit_tokens)
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
    parser.add_argument("--json-file", default="", type=str, required=True)
    parser.add_argument("--base-dir", default="", type=str, required=True)
    args = parser.parse_args()

    JSON_FILE = args.json_file
    BASE_DIR = args.base_dir

    dataset = PoseDataset(JSON_FILE, BASE_DIR)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    model = PoseEstimationModule()
    trainer = Trainer(max_epochs=10)  
    trainer.fit(model, dataloader)
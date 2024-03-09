#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   resnet_transfomer_concat.py
@Time    :   2024/03/08 16:49:38
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Pose estimation model using a combination of ResNet, Vision Transformer (ViT), and SMPL (a 3D human body model).
'''


import torch
from lightning import LightningModule
import torch.nn as nn
from transformers import ViTModel
from torchvision.models import resnet18, ResNet18_Weights
import smplx


class PoseEstimationModule(LightningModule):
    def __init__(
        self,
        learning_rate=1e-4,
        smpl_model_path=None,
        vit_name="google/vit-base-patch16-224",
    ):
        super().__init__()
        resnet_weights = ResNet18_Weights.IMAGENET1K_V1
        self.resnet = resnet18(weights=resnet_weights)
        self.resnet_out_features = self.resnet.fc.in_features

        # Remove the fully connected layer from ResNet
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.vit = ViTModel.from_pretrained(vit_name)
        self.vit_out_features = self.vit.config.hidden_size

        self.output_head = nn.Sequential(
            nn.Linear(self.resnet_out_features + self.vit_out_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 72),  # 72 pose parameters for SMPL
        )
        self.learning_rate = learning_rate

        # Initialize the SMPL model
        self.smpl = smplx.create(smpl_model_path, model_type="smpl")

        # Projection layer to map ResNet features to the desired dimension
        self.resnet_projection = nn.Linear(
            self.resnet_out_features,
            self.vit.config.hidden_size
            * (self.vit.config.image_size // self.vit.config.patch_size) ** 2,
        )

    def forward(self, images):
        batch_size = images.size(0)

        # Extract features using the ResNet
        resnet_features = self.resnet(images)
        resnet_features = resnet_features.view(batch_size, -1)

        vit_outputs = self.vit(pixel_values=images)
        vit_features = vit_outputs.last_hidden_state[:, 0, :]

        combined_features = torch.cat((resnet_features, vit_features), dim=1)
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def calculate_loss(self, output_vertices, projected_vertices, target_vertices):
        if projected_vertices.dim() == 4 and projected_vertices.size(2) == 1:
            projected_vertices = projected_vertices.squeeze(2)

        # Now calculate vertex loss and projection loss
        vertex_loss = nn.MSELoss(reduction="mean")(output_vertices, target_vertices)
        projection_loss = nn.MSELoss(reduction="mean")(output_vertices[:, :, :2], projected_vertices)
        return vertex_loss + projection_loss

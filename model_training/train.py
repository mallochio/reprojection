#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   train.py
@Time    :   2024/03/08 18:29:35
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Training code for new pose estimation models
"""
import os
import json
import torch
import numpy as np
from PIL import Image
import trimesh
import argparse
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from resnet_transfomer_concat import PoseEstimationModule
import torchvision.transforms as T

torch.set_float32_matmul_precision("high")  # or 'medium'


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
        target_vertices = pad_sequence(
            target_vertices, batch_first=True, padding_value=0
        )
        return images, projected_vertices, target_vertices


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
    main()

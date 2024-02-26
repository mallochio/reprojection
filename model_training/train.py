#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2024/02/09 19:28:47
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Script to train the model using torch
'''

import torch
import lightning as pl
from lightning.fabric import Fabric

# TODO - This is a stub, need to implement the actual dataloader
# Make a dataloader
def make_dataloader():
    # Create a dataset
    dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 10), torch.randint(0, 2, (100, ))
    )

    # Create a dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True
    )

    return dataloader

if __name__ == "__main__":
    # Create a model
    model = Fabric()

    # Create a trainer
    trainer = pl.Trainer()

    # Train the model
    trainer.fit(model)

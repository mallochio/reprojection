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


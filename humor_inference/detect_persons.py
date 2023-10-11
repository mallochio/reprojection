#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   Untitled-1
@Time    :   2023/05/08 18:24:06
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Detect persons in image using densepose and binary classify into folders.
'''

import argparse



def main(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect persons in image using densepose and binary classify into folders.')
    parser.add_argument('--input', '-i', type=str, help='Input image folder')
    parser.add_argument('--output', '-o', type=str, help='Output folder')
    args = parser.parse_args()
    main(args)


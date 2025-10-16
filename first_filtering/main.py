# Copyright 2025 AIT Austrian Institute of Technology GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Author: Miguel Castells

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import random
import concurrent.futures
import torchvision.transforms.functional as Fvis
import json
import argparse

from tqdm.auto import tqdm
from skimage.exposure import match_histograms
from torch.utils import data as data
from PIL import Image
from pathlib import Path
from .black_pixels import condition_border_bp_path, condition_border_bp_tensor

def naip_has_bp(path: Path):
    '''
    Return True if the image has more than threshold of black pixels in the borders.
    '''
    return condition_border_bp_path(str(path), 1, 2)  # condition function may expect str

def tci_exist(naip_path: Path, s2_dir: Path):
    '''
    Return True if the tci path exists.
    '''
    chip = naip_path.parent.name
    s2_path = s2_dir / chip / 'tci.png'
    return s2_path.exists()

def compute_valid_index(naip_path: Path, s2_dir: Path):
    '''
    Return (unique_id/chip, valid_index) where the valid index are the indices of the lr images
    where there are no black pixels on the borders.
    '''
    chip = naip_path.parent.name
    unique_id = naip_path.parents[1].name
    lr_path = s2_dir / chip / 'tci.png'

    lr_tensor = Fvis.pil_to_tensor(Image.open(lr_path))
    lrs_rgb = torch.reshape(lr_tensor, (3, -1, 32, 32)).permute(1, 0, 2, 3)
    valid_index = []

    for i, lr in enumerate(lrs_rgb):
        if not condition_border_bp_tensor(lr, 1, 2):
            valid_index.append(i)

    key = str(Path(unique_id) / chip)
    return (key, valid_index)

def valid_datapoints(path: Path, s2_dir: Path):
    '''
    Apply 3-step filtering on a NAIP image path. Returns (key, valid_index).
    '''
    chip = path.parent.name
    unique_id = path.parents[1].name
    key = str(Path(unique_id) / chip)

    if naip_has_bp(path):
        return (key, [])

    if not tci_exist(path, s2_dir):
        return (key, [])

    return compute_valid_index(path, s2_dir)

if __name__ == '__main__':
    # Parse command-line argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to the JSON config file')
    args = parser.parse_args()

    # Load config file
    with open(args.opt, 'r') as f:
        config = json.load(f)

    # Read parameters from config
    naip_dir = Path(config['naip_dir'])
    s2_dir = Path(config['s2_dir'])
    results_path = Path(config['results_path'])

    naip_paths = list(naip_dir.rglob('rgb.png'))
    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(valid_datapoints, p, s2_dir) for p in naip_paths]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            key, valid_index = future.result()
            if valid_index:
                results[key] = valid_index

    print(len(results))

    # Save with string keys
    with open(results_path, 'w') as file:
        json.dump(results, file, indent=2)

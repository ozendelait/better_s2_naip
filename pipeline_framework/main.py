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

import torch
import json 
import copy
import torch.nn.functional as F
import os
import concurrent.futures
import argparse

from PIL import Image
from tqdm.auto import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from .utils import load_images, histogram_matching, histogram_matching_tensors, save_img

def step_1_cloud_removal(hr, lrs, num_imgs_m1, device):
    '''
    Performs a naive cloud-removal step by selecting the `num_imgs_m1` best-matching low-resolution (LR)
    images using Peak Signal-to-Noise Ratio (PSNR) between bicubic-downsampled HR and LR images.

    Args:
        hr (torch.Tensor): High-resolution image tensor (C, H, W).
        lrs (torch.Tensor): Low-resolution image tensor batch (N, C, H, W).
        num_imgs_m1 (int): Number of LR patches to keep.
        device (str): Device string ("cuda" or "cpu").

    Returns:
        torch.Tensor: Indices of the top `num_imgs_m1` LR images.
    '''
    lrs = lrs.to(device)
    num_lr = lrs.shape[0]
    hr_res = F.interpolate(hr.unsqueeze(0).float(), (32,32), mode='bicubic').to(device)
    hrs_res = hr_res.expand((num_lr, 4, 32, 32))

    if num_imgs_m1 < num_lr:
        psnr = PeakSignalNoiseRatio(reduction=None, dim=[1,2,3], data_range=255).to(device)
        scores = psnr(lrs, hrs_res)
        indices = torch.argsort(scores, descending=True)[:num_imgs_m1]
    else:
        indices = torch.arange(num_lr, dtype=torch.uint8)

    return indices.cpu()

def step_2_metric(lrs, hr, device):
    '''
    Computes the Structural Similarity Index Measure (SSIM) between HR and LR images,
    along with a histogram-matched HR image. Selects the best LR patch based on a weighted score.

    Args:
        lrs (torch.Tensor): Filtered LR image batch (N, C, H, W).
        hr (torch.Tensor): High-resolution image (C, H, W).
        device (str): Device string.

    Returns:
        int: Index of the best-matching LR image patch.
    '''
    num_lr = lrs.shape[0]
    hr_res = F.interpolate(hr.unsqueeze(0).float(), (32,32), mode='bicubic').to(device)
    hrs_res = hr_res.expand((num_lr, 4, 32, 32))   
    hrs_res_harm = histogram_matching(hr, lrs).to(device)
    lrs = lrs.to(device)

    ssim = StructuralSimilarityIndexMeasure(data_range=255., reduction=None, gaussian_kernel=True, sigma=1.5).to(device)

    ssim_hr_lr = ssim(lrs.float(), hrs_res.float())
    ssim_hr_hrharm = ssim(hrs_res.float(), hrs_res_harm.float())

    weighted_scores = 0.7 * ssim_hr_lr + 0.3 * ssim_hr_hrharm
    best_indice = torch.argmax(weighted_scores)

    return best_indice.cpu().item()

def pipeline(unique_id_chip, valid_index, num_imgs_m1, rgb_dir, nir_dir, device):
    '''
    Executes the two-step selection pipeline:
    1. Cloud-removal using PSNR.
    2. Metric-based selection using SSIM and histogram matching.

    Args:
        unique_id_chip (str): ID of the image chip in "unique_id/chip" format.
        valid_index (list): List of valid indices of LR patches.
        num_imgs_m1 (int): Number of LR patches to keep after step 1.
        rgb_dir (str): Directory containing RGB images.
        nir_dir (str): Directory containing NIR images.
        device (str): Device to use for computation.

    Returns:
        int: Selected best LR patch index (from original set).
    '''
    # Load images
    hr, lrs = load_images(unique_id_chip, valid_index, rgb_dir, nir_dir)

    # Step 1 : naive cloud removal
    indices_step_1 = step_1_cloud_removal(hr, lrs, num_imgs_m1, device)
    lrs_step2 = lrs[indices_step_1]

    # Step 2 : Compute best LR indice 
    indice_step_2 = step_2_metric(lrs_step2, hr, device)

    return indices_step_1[indice_step_2].item()

def main(opt_path):
    '''
    Main pipeline entry point. Loads configuration from a JSON file,
    executes the image selection pipeline in parallel, and saves output images
    and tracker JSON with selected indices.

    Args:
        opt_path (str): Path to the configuration JSON file.

    Returns:
        None
    '''
    # Charger la config
    with open(opt_path, 'r') as f:
        opts = json.load(f)
    
    rgb_dir = opts['rgb_dir']
    save_dir = opts['save_dir']
    nir_dir = opts['nir_dir']
    data_path = opts['data_path']
    num_imgs_m1 = opts['num_imgs_m1']
    
    device = opts.get('device', "cuda" if torch.cuda.is_available() else "cpu")  

    tracker = {}
    tracker_path = os.path.join(save_dir, 'tracker.json')

    with open(data_path, 'r') as file:
        data = json.load(file)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_pipeline = {
            executor.submit(pipeline, unique_id_chip, valid_index, num_imgs_m1, rgb_dir, nir_dir, device): unique_id_chip 
            for unique_id_chip, valid_index in data.items()
            }
        for future in tqdm(concurrent.futures.as_completed(future_to_pipeline), total=len(future_to_pipeline)):
            unique_id_chip = future_to_pipeline[future]
            indice = future.result()
            tracker = save_img(unique_id_chip, save_dir, rgb_dir, nir_dir, indice, tracker)     

    with open(tracker_path, 'w') as file:
        json.dump(tracker, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', required=True, help='Path to config file')
    args = parser.parse_args()
    main(args.opt)






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
import torchvision
import torchvision.transforms.functional as Fvis
import torch.nn.functional as F
import os
import numpy as np

from skimage.exposure import match_histograms
from PIL import Image

def load_images(unique_id_chip, valid_index, rgb_dir, nir_dir):
    '''
    Loads the high-resolution (NAIP) and low-resolution (Sentinel-2) images for both RGB and NIR channels,
    given a unique ID and chip identifier.

    Args:
        unique_id_chip (str): A string formatted as "<unique_id>/<chip_coord>".
        valid_index (list): List of valid LR indices to extract from the LR tensor.
        rgb_dir (str): Root directory containing NAIP and Sentinel-2 RGB images.
        nir_dir (str): Root directory containing NAIP NIR images.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the high-resolution image tensor (HR)
        and the selected low-resolution image tensors (LRS), both with RGB + NIR channels.
    '''
    unique_id, chip_coord = unique_id_chip.split('/')[0], unique_id_chip.split('/')[1]
    hr_path = os.path.join(rgb_dir, 'naip', unique_id, chip_coord, 'rgb.png')
    lr_path = os.path.join(rgb_dir, 'sentinel2', chip_coord, 'tci.png')

    nir_lr_path = str(lr_path.replace('tci.png', 'b08.png'))
    lr_nir = Fvis.pil_to_tensor(Image.open(nir_lr_path))
    lrs_nir = torch.reshape(lr_nir, (-1, 32,32)).unsqueeze(0).permute(1,0,2,3)[valid_index]

    nir_hr_path = os.path.join(nir_dir, 'naip', unique_id, chip_coord, 'nir.png')
    hr_nir = Fvis.pil_to_tensor(Image.open(nir_hr_path)) 

    hr_rgb = Fvis.pil_to_tensor(Image.open(hr_path))
    lr_tensor = Fvis.pil_to_tensor(Image.open(lr_path))
    lrs_rgb = torch.reshape(lr_tensor, (3,-1, 32,32)).permute(1,0,2,3)[valid_index]

    lrs = torch.cat([lrs_rgb, lrs_nir], dim=1)
    hr = torch.cat([hr_rgb, hr_nir], dim=0)

    return hr, lrs

def histogram_matching_tensors(img: torch.Tensor, ref: torch.Tensor):
    '''
    Applies histogram matching between two image tensors.

    Args:
        img (torch.Tensor): The source image tensor to be transformed (C, H, W).
        ref (torch.Tensor): The reference image tensor to match histograms to (C, H, W).

    Returns:
        torch.Tensor: The histogram-matched version of `img`, aligned to `ref`.
    '''
    img_arr = img.permute(1,2,0).numpy()
    ref_arr = ref.permute(1,2,0).numpy()
    result = match_histograms(img_arr, ref_arr, channel_axis=-1)
    return torch.Tensor(result).permute(2,0,1)

def histogram_matching(hr: torch.Tensor, lrs_ref: torch.Tensor ):
    '''
    Resizes the high-resolution image to match the resolution of the LR images and performs
    histogram matching for each LR image individually.

    Args:
        hr (torch.Tensor): The high-resolution image tensor (C, H, W).
        lrs_ref (torch.Tensor): A batch of LR image tensors to match to (N, C, H, W).

    Returns:
        torch.Tensor: Histogram-matched HR images with shape matching lrs_ref.
    '''
    hr_res = F.interpolate(hr.unsqueeze(0).float(), (32,32), mode='bicubic')[0]
    hrs_res_harm = torch.zeros_like(lrs_ref)
    for i, lr_img in enumerate(lrs_ref):
        hrs_res_harm[i] = histogram_matching_tensors(hr_res, lr_img)
    return hrs_res_harm

def save_img(unique_id_chip, save_dir, rgb_dir, nir_dir, best_indice, json_tracker):
    '''
    Saves the histogram-matched HR and LR images for both RGB and NIR channels to disk,
    and updates a JSON tracker with the selected LR index.

    Args:
        unique_id_chip (str): A string in the format "<unique_id>/<chip_coord>".
        save_dir (str): Directory to save output images.
        rgb_dir (str): Directory containing original RGB images.
        nir_dir (str): Directory containing NIR images.
        best_indice (int): Index of the best matching LR image patch.
        json_tracker (dict): Dictionary to track selected indices.

    Returns:
        dict: Updated json_tracker with the current chip and best index.
    '''
    hr, best_lr = load_images(unique_id_chip, [best_indice], rgb_dir, nir_dir)
    best_lr = best_lr[0]
    hr_matched = histogram_matching_tensors(hr, best_lr)

    # Move back to CPU and normalize to [0,1]
    hr_to_save = (hr_matched.float()) / 255.
    lr_to_save = (best_lr.float()) / 255.

    naip_save_dir = os.path.join(save_dir, 'naip')
    s2_save_dir = os.path.join(save_dir, 'sentinel2')

    # For NAIP RGB image 
    unique_id, chip_coord = unique_id_chip.split('/')[-2], unique_id_chip.split('/')[-1]
    hr_img_dir = os.path.join(naip_save_dir, unique_id, chip_coord)
    os.makedirs(hr_img_dir, exist_ok=True)
    hr_img_path = os.path.join(hr_img_dir, 'rgb.png')
    torchvision.utils.save_image(hr_to_save[:3], hr_img_path) 

    # For S2 TCI image
    lr_img_dir = os.path.join(s2_save_dir, chip_coord)
    os.makedirs(lr_img_dir, exist_ok=True)
    lr_img_path = os.path.join(lr_img_dir, 'tci.png')
    torchvision.utils.save_image(lr_to_save[:3], lr_img_path)

    # NIR save
    # For histogram matched NAIP NIR band
    hr_nir_dir = os.path.join(naip_save_dir, unique_id, chip_coord)
    os.makedirs(hr_nir_dir, exist_ok=True)  
    hr_nir_img_path = os.path.join(hr_nir_dir, 'nir.png')
    torchvision.utils.save_image(hr_to_save[3:4], hr_nir_img_path)

    # For S2 NIR band
    lr_nir_dir = os.path.join(s2_save_dir, chip_coord)
    os.makedirs(lr_nir_dir, exist_ok=True)
    lr_nir_img_path = os.path.join(lr_nir_dir, 'nir.png')
    torchvision.utils.save_image(lr_to_save[3:4], lr_nir_img_path)

    json_tracker[str(os.path.join(unique_id, chip_coord))] = best_indice
    return json_tracker
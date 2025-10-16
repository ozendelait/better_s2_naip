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
# Author: Oliver Zendel

import gzip
import os
import json
import time
import argparse
import glob
import sys

curr_nir_script_path = os.path.dirname(os.path.realpath(__file__))
with gzip.open(curr_nir_script_path+'/sids_naip.json.gz', "r") as f:
    sids_naip = json.load(f) #mapping partial naip id from satlas file path to full id

from tqdm.auto import tqdm
import numpy as np
import cv2

from pystac_client import Client
import planetary_computer
import mercantile
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol, from_bounds
from rasterio.warp import reproject, Resampling

#convert webmercartor tile into WGS84 bounds
def webmercator2bbox4326(x,y,z):
    tile = mercantile.Tile(x=x, y=y, z=z)
    return  mercantile.bounds(tile) # EPSG:4326 (WGS84)

# download NAIP data from planetary_computer; roi is based on webmercator coords but result is still in EPSG:4326 (WGS84)
# padding is necessary to allow valid data also in the future transformed target data space in webmercator
def download_roi(item0, xyz, trg_filepath, pad_view=[[-0.25,-0.25],[0.25,0.25]], min_dim=1, keep_one_band=4):
    signed_item, bbox_4326 = planetary_computer.sign(item0), webmercator2bbox4326(xyz[0],xyz[1],xyz[2])
    signed_url = signed_item.assets.get("image").href
    with rasterio.Env():
        with rasterio.open(signed_url) as src:
            src_max_hw = [src.height-1, src.width-1]
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)        
            coords_trg_crs = [transformer.transform(bbox_4326.west, bbox_4326.north), transformer.transform(bbox_4326.east, bbox_4326.south)]
            roi_corners = [rowcol(src.transform, coords_trg_crs[j][0], coords_trg_crs[j][1]) for j in range(2)] #(x0,y0), (x1,y1)
            if not pad_view is None:
                roi_hw = [roi_corners[1][j]-roi_corners[0][j] for j in range(2)]
                roi_corners = [[int(roi_hw[c]*pad_view[j][1-c])+roi_corners[j][c] for c in range(2)] for j in range(2)]
            #clip coords
            roi_corners = [[max(min(src_max_hw[c], roi_corners[j][c]),0)  for c in range(2)] for j in range(2)]
            window = ((roi_corners[0][0], roi_corners[1][0]), (roi_corners[0][1], roi_corners[1][1])) # [(y0,y1), (x0,x1)]
            
            if min([window[i][1]-window[i][0]  for i in range(2)]) < min_dim:
                return
        
            roi_data = src.read(window=window) if keep_one_band is None else src.read(keep_one_band, window=window)[np.newaxis, :, :]
            meta = src.meta.copy()
            if not keep_one_band is None:
                meta.update({"count":1})
            meta.update({
                "driver": "GTiff",
                "height": roi_data.shape[-2],  # rows
                "width": roi_data.shape[-1],   # cols
                "transform": src.window_transform(window),
                "crs": src.crs
            })
            with rasterio.open(trg_filepath, "w", **meta) as dst:
                dst.write(roi_data)

#load geotiff, transform into target webmercator data; band 4 == NIR for full geotiffs; otherwise 1
def convert_roi_nir(geotiff_path, xyz, calc_bandids=[1], dst_wh=(128,128), resampling_method=Resampling.nearest ):
    left, bottom, right, top = mercantile.xy_bounds(xyz[0],xyz[1],xyz[2])
    dst_transform = from_bounds(left, bottom, right, top, dst_wh[0], dst_wh[1])
    with rasterio.Env():
        with rasterio.open(geotiff_path) as src:
            dst_array = np.zeros((src.count, dst_wh[1], dst_wh[0]), dtype=np.float32)
            for i in calc_bandids:
                reproject(  source=rasterio.band(src, i),
                            destination=dst_array[max(0,i-1)],
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=dst_transform,
                            dst_crs="EPSG:3857",  # destination is in Web Mercator
                            resampling=resampling_method    # nearest or cubic, lanczos, etc.
                        )
            dst_array = np.ascontiguousarray(np.transpose(np.uint8(np.clip(np.round(dst_array),0,255)),(1,2,0)))
            return dst_array

def get_stac_coll(root_url = "https://planetarycomputer.microsoft.com/api/stac/v1", coll="naip"):
    return Client.open(root_url, modifier=planetary_computer.sign_inplace).get_collection(coll)

def add_naip_nir(naip_coll, 
                original_rgb_path, 
                trg_rootdir=None,
                sleep_between_calls = 1.1,
                sleep_before_sign = 0.1,
                tmp_geotiff_path='./.tmp_naip_nir.tif',
                skip_existing=True):
    #in satlas paths, the subdirs correspond to NAIP ids and webmercator coordinates:
    # e.g. val_set/naip/m_4709606_ne_14_060_20190723/30475_45569/rgb.png
    # has a partial NAIP id of m_4709606_ne_14_060_20190723 and 
    # webmercator tile coordinates of x=30475, y=45569; z is always 17 for satlas
    op_spl = original_rgb_path.replace('\\','/').split('/')
    xyz = [int(c) for c in op_spl[-2].split('_')]+[17]
    if trg_rootdir is None: #put next to rgb (default)
        trg_path = '/'.join(op_spl[:-1])+'/nir.png' 
    else: #create parallel structure with same subdirs
        trg_path = f'{trg_rootdir}/{op_spl[-5]}/{op_spl[-4]}/{op_spl[-3]}/{op_spl[-2]}/nir.png'

    if skip_existing and os.path.exists(trg_path):
        return
    item0 = naip_coll.get_item(sids_naip[op_spl[-3]])
    time.sleep(sleep_before_sign) #search directly followed by signing op can be too much traffic
    download_roi(item0, xyz, tmp_geotiff_path)
    t0 = time.time()
    nir_data = convert_roi_nir(tmp_geotiff_path, xyz)
    os.makedirs(os.path.dirname(trg_path),exist_ok=True)
    cv2.imwrite(trg_path, nir_data)
    t1 = time.time()
    time_sleep_open = sleep_between_calls - (t1-t0)
    if time_sleep_open > 0: #conform to webapi; don't spam downloader. No parallel downloads supported
        time.sleep(time_sleep_open)

def main_callable(arg0, tqdm_ver=tqdm, fail_on_err_retry=4, sleep_on_err=5.0):
    # parser setup
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument('--src_set', type=str, 
                        default="/workspace/data/val_set",
                        help="Src SATLAS set dir (one above the root dir)")
    parser.add_argument('--trg', type=str, default=None,
                        help="Target NAIP NIR data output root directory; default: put into src root dir")
    args = parser.parse_args(arg0)
    print("Reading folder "+args.src_set+ " for paths; this process can take a few seconds...")
    if os.path.exists(args.src_set+'/val_set/naip/'): #actual root dir; convert both val_set and train_urban_set
        all_rgb_paths = sorted(glob.glob(args.src_set+'/val_set/naip/**/rgb.png', recursive=True))
        all_rgb_paths += sorted(glob.glob(args.src_set+'/train_urban_set/naip/**/rgb.png', recursive=True))
    else:
        all_rgb_paths = sorted(glob.glob(args.src_set+'/naip/**/rgb.png', recursive=True))
    naip_coll = get_stac_coll()
    for p in tqdm_ver(all_rgb_paths, total=len(all_rgb_paths), desc="Adding NAIP NIR data"):
        for retry_on_err in range(fail_on_err_retry):
            try:
                add_naip_nir(naip_coll, p, trg_rootdir=args.trg)
                break
            except Exception as ex:
                if retry_on_err >= fail_on_err_retry-1:
                    print("Downloader error: ", ex)
                    return
                time.sleep(sleep_on_err)
                naip_coll = get_stac_coll()

if __name__ == "__main__":
    sys.exit(main_callable(sys.argv[1:]))




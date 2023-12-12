import itertools
from tqdm import tqdm

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
import shapely
from pathlib import Path
import matplotlib.pyplot as plt
import sys
PATCH_SIZE=1024
PATCH_STRIDE = PATCH_SIZE
DATA_DIR=Path('data/surface_water/oberbayern/')
DATA='Sentinel-2/Oberbayer_10m_3035_4bands.tif'
# DATA='Sentinel-2/test_sentinel.tif'
MASKS = "oberbayern-water-mask.tif"

from imageio import imwrite

rio_gdal_options = {
    'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif,.xml',  # Adding .xml is essential!
}

def normalize_layer(data:np.array,min:float=None, max:float=None):
    if not min:
        min = np.min(data)
    if not max:
        max = np.max(data)
    data=((data-min)/(max-min))
    data[data<0]=0
    data[data>1]=1
    return data

def pad_images(rgb,expected_shape=(1024,1024,3), constant_value=0):
        if rgb.shape != expected_shape:
            pad_width=[(0,expected_shape[i]-rgb.shape[i]) for i in range(len(rgb.shape))]
            return np.pad(rgb,pad_width=pad_width, mode='constant', constant_values=constant_value)
        else:
            return rgb

def generate_tiles(
    size=PATCH_SIZE,
    stride=PATCH_STRIDE,
    tiles_path=DATA_DIR.joinpath(f"png_new"),
    regenerate=False,
    cumulative_count_cut=False
):
    print(f"Start generating tiles at the path {tiles_path}")
    if tiles_path.exists() and not regenerate:
        print("Tiles folder already exists. It's assumed that tiles do not need to be generated. If you still want to regenerate the tiles use the keyword regenerate=True with the function.")
        return tiles_path

    src = rasterio.open(DATA_DIR.joinpath(DATA))
    masks = rasterio.open(DATA_DIR.joinpath(MASKS))
    plan = list(
        itertools.product(range(0, src.width, stride), range(0, src.height, stride))
    )
    if not cumulative_count_cut:
        red_min = np.min(src.read(3).astype(float))
        red_max = np.max(src.read(3).astype(float))
        green_min = np.min(src.read(2).astype(float))
        green_max = np.max(src.read(2).astype(float))
        blue_min = np.min(src.read(1).astype(float))
        blue_max = np.max(src.read(1).astype(float))

    print(f"Name: {src.name}, width: {src.width}, height: {src.height}, crs: {src.crs}, indexes: {src.indexes}")
    print(f"Name: {masks.name}, width: {masks.width}, height: {masks.height}, crs: {masks.crs}, indexes: {masks.indexes}")
    for w, h in tqdm(plan):
        window = Window(w, h, size, size)
        red = src.read(3, window=window).astype(float)
        green = src.read(2, window = window).astype(float)
        blue = src.read(1, window = window).astype(float)
        rgb_original = rgb= np.stack((red, green, blue),axis=-1)
        if cumulative_count_cut:
        # Cumulative count cut for values 2-98%
            red=normalize_layer(red, 162,1586)
            green=normalize_layer(green, 274,1444)
            blue=normalize_layer(blue, 156, 1198)
        else:
        # Without cumulative count cut but overall normlization per color:
            red=normalize_layer(red, red_min, red_max)
            green=normalize_layer(green, green_min, green_max)
            blue=normalize_layer(blue, blue_min, blue_max)
        rgb= np.stack((red, green, blue),axis=-1)
        rgb = (rgb*255).astype(np.uint8)
        rgb = pad_images(rgb)
        imwrite(tiles_path.joinpath(f'patch-{w}-{h}-image.png'),rgb)

        label=masks.read(1, window=window)
        # remove all labels from masks where there is no image
        label=(np.any(rgb_original !=[0,0,0], axis=-1)*label)
        label=np.logical_not(label)
        label=(label*255).astype(np.uint8)
        label=pad_images(label, (1024,1024,1), 255)
        imwrite(tiles_path.joinpath(f'patch-{w}-{h}-osm.png'), label)
    return tiles_path


if __name__ == '__main__':
    generate_tiles(regenerate=True, cumulative_count_cut=True)
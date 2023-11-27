import cv2
import numpy as np
import os
import random
import shapely

from PIL import Image
from rasterio import features
from torch.utils.data import Dataset


def random_points_in_polygon(polygon, nr_pts):
    pts = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(pts) < nr_pts:
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        if polygon.contains(shapely.Point(x, y)):
            pts.append([x, y])
    return pts


def mask_to_prompt(mask_array, prompt_type, nr_pts):
    mask = mask_array == 0
    shapes = features.shapes(mask_array, mask)
    polygons = [shapely.geometry.shape(shape) for shape, _ in shapes]

    if polygons:
        if prompt_type=='bb':
            bb = shapely.envelope(polygons)
            return [list(box.bounds) for box in bb]

        if prompt_type=='center_pt':
            # centroid doesnt return a point that lies inside the polygon
            # for irregular shapes. representative_point cheaply computes a
            # point that always lies within the polygon
            pts = [polygon.representative_point() for polygon in polygons]
            return [[[pt.x, pt.y]] for pt in pts]

        if prompt_type=='multiple_pts':
            return [random_points_in_polygon(polygon, nr_pts) for polygon in polygons]

        if prompt_type=='foreground_background_pts':
            foreground_pts = random.sample(np.argwhere(mask==0).tolist(), nr_pts)
            background_pts = random.sample(np.argwhere(mask==255).tolist(), nr_pts)
            return [foreground_pts, background_pts]

    else:
        return []


class GeoDataset(Dataset):
    def __init__(self, img_dir, img_list, prompt_type, nr_pts, out_dir):
        self.img_dir = img_dir
        self.img_list = img_list
        self.prompt_type = prompt_type
        self.nr_pts = nr_pts
        self.out_dir = out_dir

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # get mask and generate its prompts then transform it into binary mask
        # the prompts are generated before to account for every building on its own
        # (there is a color fading between the outlines of the buildings)
        mask = cv2.imread(os.path.join(self.img_dir, self.img_list[idx][:-9] + 'osm.png'), 0).astype(np.uint8)
        prompts = mask_to_prompt(mask, self.prompt_type, self.nr_pts)
        mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY_INV)[1]

        sample = {
            'image' : Image.open(os.path.join(self.img_dir, self.img_list[idx])),
            'gt' : mask,
            'prompts': prompts,
            'pred_file': os.path.join(self.out_dir, self.img_list[idx][:-9] + 'pred.png')
        }
        return sample

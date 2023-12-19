from imageio.v2 import imwrite, imread
import numpy as np
import glob
from pathlib import Path
from rasterio import features
import shapely
from tqdm import tqdm
import pandas as pd
import csv
import cv2
DATA_PATH = [ "data/bbd1k", "data/water1k"]
KEYWORD_MASKS = 'osm'


if __name__ == '__main__':
    for path in DATA_PATH:
        print(f"ANALYZE {path}")
        labels_per_image=[]
        label_density=[]
        label_size=[]
        bb_size=[]
        bb_label_relation=[]
        for file in glob.iglob(path+"/*split*"):
            # get files which are not filtered in the split list
            split_list = pd.read_csv(file)
            img_list = split_list.filename[split_list.detections==True]
            label_list=[img.split("-image", maxsplit=1)[0]+'-osm.png' for img in img_list]
            print(f'total images filtered: {len(img_list)}/{len(split_list)}, Amount of labels to calc statistics on: {len(label_list)}')
        for file in tqdm(label_list):
            label= cv2.imread(path+'/'+file, 0).astype(np.uint8)
            mask = label == 0 # as labels are inverse in our data, that means interesting areas are black
            label_density.append(np.count_nonzero(mask)/(mask.shape[0]*mask.shape[1]))
            shapes = features.shapes(label, mask)
            polygons = [shapely.geometry.shape(shape) for shape, _ in shapes]
            labels_per_image.append(len(polygons))
            areas = shapely.area(polygons)
            label_size.extend(np.ndarray.tolist(areas))
            bb=[list(box.bounds) for box in shapely.envelope(polygons)]
            bb_areas = [(x[2]-x[0]) * (x[3]-x[1]) for x in bb]
            bb_size.extend(bb_areas)
            bb_label_relation.extend([i / j for i, j in zip(label_size, bb_areas)])

        print(f"Avg. Label Density {(np.array(label_density).mean())}, \nAvg. amount of labels per patch {np.array(labels_per_image).mean()}, \nAvg. Label Size {np.array(label_size).mean()}, \nAvg. bb size {np.array(bb_size).mean()}, \nAvg. bb label relation {np.array(bb_label_relation).mean()}")
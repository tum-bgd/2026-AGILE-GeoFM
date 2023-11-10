import argparse
import cv2
import glob
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create .csv file with data train/val/test split.')
    parser.add_argument('--data_dir', type=str, default='data/bbd1k/',  help='specify the root path of images')
    parser.add_argument('--split_file', type=str, default='bbd1k_data_split.csv',  help='specify the name of the .csv output')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='proportion of the dataset to include in the test split.')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='proportion of the dataset to include in the validation split')
    args = parser.parse_args()

    # There are 18 OSM masks without a corresponding image, that is why, I use images to extract filename list
    image_names = [os.path.basename(x) for x in glob.glob(args.data_dir + "*-image.png")]

    df = pd.DataFrame({"filename": image_names})
    df["split"] = "train"
    df["buildings"] = False

    for idx, img_name in df["filename"].items():
        image = cv2.imread(os.path.join(args.data_dir, img_name))
        # Some images (813 out of 15828) have black borders (which are as well reflected in the masks) 
        # from the tiling process, remove these from the list.
        if np.any(np.all(image==0, axis=2)):
            df.drop(idx, inplace=True)
        else:
            mask = cv2.imread(os.path.join(args.data_dir, img_name[:-9] + 'osm.png'), 0).astype(np.uint8)
            if not np.all(mask==255):
                df.loc[idx, "buildings"] = True

    train_idx, test_idx = train_test_split(df.index, test_size=args.test_ratio, stratify=df.buildings)
    df.loc[test_idx, "split"] = "test"

    train_idx, val_idx = train_test_split(train_idx, test_size=args.val_ratio/(1.0-args.test_ratio), stratify=df.loc[train_idx, "buildings"])
    df.loc[val_idx, "split"] = "val"

    df.to_csv(args.data_dir + args.split_file, index=False)

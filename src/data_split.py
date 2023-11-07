import argparse
import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create .csv file with data train/val/test split.')
    parser.add_argument('--data_dir', type=str, default='data/bbd1k/',  help='specify the root path of images')
    parser.add_argument('--csv_dir', type=str, default='bbd1k_data_split.csv',  help='specify the path of the .csv output')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='proportion of the dataset to include in the test split.')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='proportion of the dataset to include in the validation split')
    args = parser.parse_args()

    image_names = [os.path.basename(x) for x in glob.glob(args.data_dir + "*-image.png")]

    df = pd.DataFrame({"filename": image_names})
    df["split"] = "train"

    train_idx, test_idx = train_test_split(df.index, test_size=args.test_ratio)
    df.loc[test_idx, "split"] = "test"

    train_idx, val_idx = train_test_split(train_idx, test_size=args.val_ratio/(1.0-args.test_ratio))
    df.loc[val_idx, "split"] = "val"

    df.to_csv(args.data_dir + args.csv_dir, index=False)

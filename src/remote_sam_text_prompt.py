import argparse
import cv2
import numpy as np
import os
import pandas as pd
import torch

from tqdm import tqdm

from src.evaluation import Evaluation
from src.visualisation import Visualizer

from RemoteSAM.args import get_parser
from RemoteSAM.lib import segmentation
from RemoteSAM.tasks.code.model import RemoteSAM


def main(args):
    dataset = args.dataset
    img_dir = f'data/{dataset}/'
    filter_file = f'data/{dataset}/{dataset}_data_filtered.csv'
    remote_sam_checkpoint = args.remote_sam_checkpoint
    text_prompt = args.text_prompt
    out_dir = f'{args.out_dir}/{dataset}/remote_sam_text_prompt/{"_".join(text_prompt)}/'

    # Create masks output folder
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Get test images filenames of images containing detections
    filter_list = pd.read_csv(filter_file)
    img_list = filter_list.filename[filter_list.detections==True]
    print(f'total images: {len(img_list)}')

    # Load RemoteSAM Model
    args2, _ = get_parser().parse_known_args()
    args2.device = "cuda" if torch.cuda.is_available() else "cpu"
    args2.window12 = True
    args2.ck_bert = "./weights/bert-base-uncased/"

    model = segmentation.__dict__["lavt_one"](pretrained='', args=args2)
    model.load_state_dict(torch.load(remote_sam_checkpoint, map_location='cpu')['model'], strict=False)
    model = model.to(args2.device)

    model = RemoteSAM(model, args2.device, use_EPOC=False)

    # Instantiate the Evaluator and Visualizer
    evaluation = Evaluation()
    visualizer = Visualizer('remote_sam', text_prompt)

    # RemoteSAM expects 896x896 images, tile each image in 4
    patch_size = 896
    img_size = 1024
    stride = img_size - patch_size
    coords = [(0, 0), (stride, 0), (0, stride), (stride, stride)]

    # Prediction
    for _, img_name in enumerate(tqdm(img_list)):
        image = cv2.imread(os.path.join(img_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # transform mask into binary a mask
        # (there is a color fading between the outlines of the buildings)
        gt_mask = cv2.imread(os.path.join(img_dir, img_name[:-9] + 'osm.png'), 0).astype(np.uint8)
        gt_mask = cv2.threshold(gt_mask, 127, 1, cv2.THRESH_BINARY_INV)[1]

        pred_mask = np.zeros((img_size, img_size), np.uint8) 

        for (x, y) in coords:
            patch = image[y:y+patch_size, x:x+patch_size]

            mask = model.semantic_seg(image=patch, classnames=text_prompt)
            for c in text_prompt:
                pred_mask[y:y+patch_size, x:x+patch_size] |= mask[c].astype(np.uint8)

        # Save the image
        pred_name = os.path.join(out_dir, img_name[:-9] + 'pred.png')
        visualizer.save(image, None, gt_mask, pred_mask, pred_name)

        # Evaluate
        gt_mask = torch.tensor(gt_mask).unsqueeze(0)
        pred_mask = torch.tensor(pred_mask).unsqueeze(0)
        evaluation.evaluate(gt_mask, pred_mask)

    # Evaluation
    evaluation.evaluate_all(out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bbd1k', choices=['bbd1k', 'water1k'])
    parser.add_argument('--remote_sam_checkpoint', type=str, default='./weights/RemoteSAMv1.pth')
    parser.add_argument('--text_prompt', type=str, nargs='+', default=['building'], help='format for multiple text prompts: chair person dog')
    parser.add_argument('--out_dir', type=str, default='results')

    args = parser.parse_args()

    main(args)

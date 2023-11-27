import argparse
import cv2
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SamProcessor, SamModel

from dataset import GeoDataset
from evaluation import Evaluation
from visualisation import Visualizer


def main(args):
    img_dir = args.img_dir
    split_file = args.split_file
    model_name = 'facebook/sam-vit-' + args.model_name
    prompt_type = args.prompt
    nr_pts = args.nr_pts
    if prompt_type == 'multiple_pts' or prompt_type == 'foreground_background_pts':
        out_dir = f'{args.out_dir}{args.model_name}/{args.prompt}_{args.nr_pts}/'
    else:
        out_dir = f'{args.out_dir}{args.model_name}/{args.prompt}/'

    # Create masks output folder
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Get test images filenames of images containing buildings
    split_list = pd.read_csv(split_file)
    img_list = split_list.filename[split_list.buildings==True].tolist()

    # Create a dataloader to batch the data
    dataset = GeoDataset(img_dir, img_list, prompt_type, nr_pts)
    dataloader = DataLoader(dataset, batch_size=32, drop_last=False)
    print(f'total images: {len(dataset)}')

    # Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = SamProcessor.from_pretrained(model_name)
    model = SamModel.from_pretrained(model_name).to(device)

    # Instantiate the Evaluator and Visualizer
    evaluation = Evaluation()
    visualizer = Visualizer(prompt_type)

    # Prediction
    for batch in tqdm(dataloader):
        if prompt_type == 'bb':
            inputs = processor(batch['image'], 
                               input_boxes=batch['prompts'], 
                               return_tensors="pt").to(device)
        elif prompt_type == 'foreground_background_pts':
            inputs = processor(batch['image'], 
                               input_points=batch['prompts'], 
                               input_labels=[[1]*nr_pts, [0]*nr_pts], 
                               return_tensors="pt").to(device)
        else:
            inputs = processor(batch['image'], 
                               input_points=batch['prompts'], 
                               return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)

        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu().detach(), 
            inputs["original_sizes"].cpu(), 
            inputs["reshaped_input_sizes"].cpu()
        )

        # assemble the multiple predicted masks for the same image into one
        pred_masks = [torch.any(mask, dim=0).squeeze().type(torch.uint8) for mask in masks]

        # Save the batch of images
        for i, sample in enumerate(batch):
            visualizer.save(sample['pred_file'], pred_masks[i])

        # Evaluate the current batch
        gt_masks = torch.tensor(np.stack(batch['gt']))
        pred_masks = torch.stack(pred_masks)
        evaluation.evaluate(gt_masks, pred_masks)

    # Evaluation
    evaluation.evaluate_all(out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='data/bbd1k/',  help='specify the path of the input images')
    parser.add_argument('--split_file', type=str, default='data/bbd1k/bbd1k_data_split.csv',  help='specify the path of the .csv data split file')
    parser.add_argument('--model_name', type=str, default='large', choices=['base', 'large', 'huge'])
    parser.add_argument('--prompt', type=str, default='bb', choices=['bb', 'center_pt', 'multiple_pts', 'foreground_background_pts'])
    parser.add_argument('--nr_pts', type=int, default=20)
    parser.add_argument('--out_dir', type=str, default='models/sam_gt_prompt/')

    args = parser.parse_args()

    main(args)

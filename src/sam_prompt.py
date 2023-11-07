import argparse
import cv2
import numpy as np
import os
import pandas as pd
import shapely
import torch

from PIL import Image
from rasterio import features
from tqdm import tqdm
from transformers import SamProcessor, SamModel


def mask_to_prompt(mask_array, prompt_type):
    mask = mask_array == 0
    shapes = features.shapes(mask_array, mask)
    polygons = [shapely.geometry.shape(shape) for shape, _ in shapes]

    if polygons:
        if prompt_type=='bb':
            bb = shapely.envelope(polygons)
            bb = [[list(box.bounds) for box in bb]]
            return bb
        if prompt_type=='pts':
            pts = shapely.centroid(polygons)
            pts = [[[pt.x, pt.y]] for pt in pts]
            return pts
    else:
        return []


def masks_to_image(masks):
    mask = torch.any(masks, 0).numpy().squeeze()
    image = Image.fromarray((~mask).astype(np.uint8) * 255)
    return image


def main(args):
    model_name = args.model_name
    out_dir = args.out_dir
    prompt_type = args.prompt
    split_file = args.split_file
    img_dir = args.img_dir

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    split_list = pd.read_csv(split_file)
    test_list = split_list.filename[split_list.split == "test"]
    print(f'total: {len(test_list)}')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = SamProcessor.from_pretrained(model_name)
    model = SamModel.from_pretrained(model_name).to(device)

    for _, img_name in enumerate(tqdm(test_list)):

        image = Image.open(os.path.join(img_dir, img_name))

        gt_mask = cv2.imread(os.path.join(img_dir, img_name[:-9] + 'osm.png'), 0).astype(np.uint8)
        gt_mask[gt_mask<=128] = 0
        gt_mask[gt_mask>128] = 255
        prompt = mask_to_prompt(gt_mask, prompt_type)

        if prompt:
            inputs = processor(image, input_boxes=prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output = model(**inputs, multimask_output=False)

            masks = processor.image_processor.post_process_masks(
                output.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
                )
        else:
            masks = [torch.full((1, 1, 1024, 1024), False)]

        pred_image = masks_to_image(masks[0])
        pred_image.save(os.path.join(out_dir, img_name[:-9] + 'pred.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='facebook/sam-vit-large')
    parser.add_argument('--out_dir', type=str, default='models/sam_large/bb_prompt/')
    parser.add_argument('--prompt', type=str, default='bb')
    parser.add_argument('--split_file', type=str, default='data/bbd1k/bbd1k_data_split.csv',  help='specify the path of the .csv data split file')
    parser.add_argument('--img_dir', type=str, default='data/bbd1k/',  help='specify the path of the input images')

    args = parser.parse_args()

    main(args)

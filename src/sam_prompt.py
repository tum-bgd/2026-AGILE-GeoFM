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

from evaluation import Evaluation
from visualisation import Visualizer


def random_points_in_background(polygon):
    pts = []
    while len(pts) < nr_pts:
        x = np.random.uniform(0, 1024)
        y = np.random.uniform(0, 1024)
        if not polygon.contains(shapely.Point(x, y)):
            pts.append([x, y])
    return pts


def random_points_in_polygon(polygon):
    pts = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(pts) < nr_pts:
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        if polygon.contains(shapely.Point(x, y)):
            pts.append([x, y])
    return pts


def mask_to_prompt(mask_array):
    mask = mask_array == 0
    shapes = features.shapes(mask_array, mask)
    polygons = [shapely.geometry.shape(shape) for shape, _ in shapes]
    areas = shapely.area(polygons)

    if polygons:
        if prompt_type=='bb':
            bb = shapely.envelope(polygons)
            return [list(box.bounds) for box in bb], areas

        if prompt_type=='center_pt':
            # centroid doesnt return a point that lies inside the polygon
            # for irregular shapes. representative_point cheaply computes a
            # point that always lies within the polygon
            pts = [polygon.representative_point() for polygon in polygons]
            return [[[pt.x, pt.y]] for pt in pts], areas

        if prompt_type=='multiple_pts':
            return [random_points_in_polygon(polygon) for polygon in polygons], areas

        if prompt_type=='foreground_background_pts':
            multipolygon = shapely.MultiPolygon(polygons)
            foreground_pts = random_points_in_polygon(multipolygon)
            background_pts = random_points_in_background(multipolygon)
            return [foreground_pts + background_pts], areas

    else:
        return []


def main(args):
    global prompt_type
    global nr_pts

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
    img_list = split_list.filename[split_list.buildings==True]
    print(f'total images: {len(img_list)}')

    # Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"The device used is {device}")
    processor = SamProcessor.from_pretrained(model_name)
    model = SamModel.from_pretrained(model_name).to(device)

    # Instantiate the Evaluator and Visualizer
    evaluation = Evaluation()
    visualizer = Visualizer(prompt_type, nr_pts)

    # Prediction
    for _, img_name in enumerate(tqdm(img_list)):
        image = Image.open(os.path.join(img_dir, img_name))

        # get mask and generate its prompts then transform it into binary mask
        # the prompts are generated before to account for every building on its own
        # (there is a color fading between the outlines of the buildings)
        gt_mask = cv2.imread(os.path.join(img_dir, img_name[:-9] + 'osm.png'), 0).astype(np.uint8)
        prompts, areas = mask_to_prompt(gt_mask)
        gt_mask = cv2.threshold(gt_mask, 127, 1, cv2.THRESH_BINARY_INV)[1]

        if prompt_type == 'bb':
            inputs = processor(image,
                               input_boxes=[prompts],
                               return_tensors="pt").to(device)
        elif prompt_type == 'foreground_background_pts':
            inputs = processor(image,
                               input_points=[prompts],
                               input_labels=[[[1]*nr_pts + [0]*nr_pts]],
                               return_tensors="pt").to(device)
        else:
            inputs = processor(image,
                               input_points=[prompts],
                               return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)

        pred_mask = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu().detach(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )

        # drop the masks that are too large compared to the ground truth
        if prompt_type != 'foreground_background_pts':
            pred_mask = pred_mask[0][torch.sum(pred_mask[0], dim=(1, 2, 3)) <= 5 * torch.tensor(areas)]
        else:
            pred_mask = pred_mask[0]

        # assemble the multiple predicted masks for the same image into one
        pred_mask = torch.any(pred_mask, dim=0).type(torch.uint8)

        # Save the image
        pred_name = os.path.join(out_dir, img_name[:-9] + 'pred.png')
        visualizer.save(image, prompts, gt_mask, pred_mask[0], pred_name)

        # Evaluate the current batch
        gt_mask = torch.tensor(gt_mask).unsqueeze(0)
        evaluation.evaluate(gt_mask, pred_mask)

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

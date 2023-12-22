import argparse
import cv2
import numpy as np
import open_clip
import os
import pandas as pd
import torch

from PIL import Image
from scipy.ndimage import maximum_filter
from tqdm import tqdm
from transformers import pipeline

from evaluation import Evaluation
from visualisation import Visualizer


def crop_image(image, mask):
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    masked = image * np.expand_dims(mask, -1)
    crop = masked[y : y + h, x : x + w]
    if h > w:
        top, bottom, left, right = 0, 0, (h - w) // 2, (h - w) // 2
    else:
        top, bottom, left, right = (w - h) // 2, (w - h) // 2, 0, 0
    # padding
    crop = cv2.copyMakeBorder(
        crop,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    crop = Image.fromarray(crop)
    return crop


def main(args):
    dataset = args.dataset
    img_dir = f'data/{dataset}/'
    filter_file = f'data/{dataset}/{dataset}_data_filtered.csv'
    model_name = 'facebook/sam-vit-' + args.model_name
    points_per_side = args.points_per_side
    clip_model_name = args.clip_model_name
    label = args.label
    clip_threshold = args.clip_threshold
    out_dir = f'{args.out_dir}/{dataset}/sam_auto_prompt/{args.clip_model_name}/{args.points_per_side}-{args.clip_threshold}'

    # Create masks output folder
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Get test images filenames of images containing detections
    filter_list = pd.read_csv(filter_file)
    img_list = filter_list.filename[filter_list.detections==True]
    print(f'total images: {len(img_list)}')

    # Load SAM Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"The device used is {device}")
    generator = pipeline("mask-generation", model=model_name, device=device)

    # Load OpenClip
    clip_model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name)
    tokenizer = open_clip.get_tokenizer(clip_model_name)
    ckpt = torch.load(f"weights/RemoteCLIP-{clip_model_name}.pt", map_location="cpu")
    message = clip_model.load_state_dict(ckpt)
    print(message)
    clip_model = clip_model.cuda().eval()

    text_queries = [f"satellite image of a {label}", "satellite image of background"]
    text = tokenizer(text_queries)

    # Instantiate the Evaluator and Visualizer
    evaluation = Evaluation()
    visualizer = Visualizer('auto_sam_classified', None)

    # Prediction
    for _, img_name in enumerate(tqdm(img_list)):
        image = Image.open(os.path.join(img_dir, img_name))

        # transform mask into binary a mask
        # (there is a color fading between the outlines of the buildings)
        gt_mask = cv2.imread(os.path.join(img_dir, img_name[:-9] + 'osm.png'), 0).astype(np.uint8)
        gt_mask = cv2.threshold(gt_mask, 127, 1, cv2.THRESH_BINARY_INV)[1]

        # generate all the masks with SAM automatic
        outputs = generator(image, points_per_crop=points_per_side, points_per_batch=1024)
        masks = outputs["masks"]

        # classify the generated masks
        pos_masks = []
        for mask in masks:
            mask_buffered = maximum_filter(mask, size=11, mode='constant', cval=0)>0
            input = crop_image(np.array(image), mask_buffered)

            mask_processed = preprocess(input).unsqueeze(0)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = clip_model.encode_image(mask_processed.to(device))
                text_features = clip_model.encode_text(text.to(device))

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]

            if text_probs[0] >= clip_threshold:
                pos_masks.append(torch.tensor(mask))

        # assemble the multiple predicted and positively classified masks into one
        if pos_masks:
            pred_mask = torch.stack(pos_masks, dim=0)
            pred_mask = torch.any(pred_mask, dim=0, keepdim=True).type(torch.uint8)
        else:
            pred_mask = torch.zeros((1, 1024, 1024), dtype=torch.uint8)

        # Save the image
        pred_name = os.path.join(out_dir, img_name[:-9] + 'pred.png')
        visualizer.save(image, masks, gt_mask, pred_mask[0], pred_name)

        # Evaluate
        gt_mask = torch.tensor(gt_mask).unsqueeze(0)
        evaluation.evaluate(gt_mask, pred_mask)

    # Evaluation
    evaluation.evaluate_all(out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bbd1k')
    parser.add_argument('--model_name', type=str, default='large', choices=['base', 'large', 'huge'])
    parser.add_argument('--points_per_side', type=int, default='128', help='number of points per side for grid sampling with SAM automatic')
    parser.add_argument('--clip_model_name', type=str, default='ViT-B-32', choices=['ViT-B-32', 'ViT-L-14'])
    parser.add_argument('--clip_threshold', type=float, default='0.7')
    parser.add_argument('--label', type=str, default='building', choices=['building', 'surface water'])
    parser.add_argument('--out_dir', type=str, default='results/')

    args = parser.parse_args()

    main(args)

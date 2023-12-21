import argparse
import cv2
import numpy as np
import os
import pandas as pd
import torch
import torchvision.transforms as T

from PIL import Image
from tqdm import tqdm
from groundingdino.util.inference import load_model, predict
from torchvision.ops import box_convert
from transformers import SamProcessor, SamModel

from evaluation import Evaluation
from visualisation import Visualizer


def main(args):
    dataset = args.dataset
    img_dir = f'data/{dataset}/'
    split_file = f'data/{dataset}/{dataset}_data_split.csv'
    model_name = 'facebook/sam-vit-' + args.model_name
    dino_model_name = args.dino_model_name
    text_prompt = args.text_prompt
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    out_dir = f'{args.out_dir}/{dataset}/sam_dino_prompt/{args.dino_model_name}/{args.text_prompt}/'

    # Create masks output folder
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Get test images filenames of images containing detections
    split_list = pd.read_csv(split_file)
    img_list = split_list.filename[split_list.detections==True]
    print(f'total images: {len(img_list)}')

    # Load SAM Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"The device used is {device}")
    processor = SamProcessor.from_pretrained(model_name)
    model = SamModel.from_pretrained(model_name).to(device)

    # Load GroundingDINO Model
    if dino_model_name == 'SwinT':
        config_path = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        checkpoint_path = "./weights/groundingdino_swint_ogc.pth"
    else:
        config_path = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
        checkpoint_path = "./weights/groundingdino_swinb_cogcoor.pth"
    dino_model = load_model(model_config_path=config_path, model_checkpoint_path=checkpoint_path)

    # Transform image to tensor
    transform = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Instantiate the Evaluator and Visualizer
    evaluation = Evaluation()
    visualizer = Visualizer('text_prompt', text_prompt)

    # Prediction
    for _, img_name in enumerate(tqdm(img_list)):
        image = Image.open(os.path.join(img_dir, img_name))
        image_tensor = transform(image)

        # generate bounding box prompts with GroundingDino to be fed into SAM
        boxes, _, _ = predict(
            model=dino_model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        # Convert box prompts from xywh to xyxy
        w = h = 1024
        boxes = boxes * torch.Tensor([w, h, w, h])
        prompts = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").tolist()

        # transform mask into binary a mask
        # (there is a color fading between the outlines of the buildings)
        gt_mask = cv2.imread(os.path.join(img_dir, img_name[:-9] + 'osm.png'), 0).astype(np.uint8)
        gt_mask = cv2.threshold(gt_mask, 127, 1, cv2.THRESH_BINARY_INV)[1]

        # In the case of bbd1k, we have the domain knowledge that no building spans the entire image patch
        # In many cases, Grounding DINO returns a bb spanning the entire image, these are therefore removed
        if dataset == 'bbd1k':
            for i, prompt in reversed(list(enumerate(prompts))):
                if (prompt[2]-prompt[0] > 950) or (prompt[3]-prompt[1] > 950):
                    prompts.pop(i)

        if prompts:
            inputs = processor(image,
                            input_boxes=[prompts],
                            return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs, multimask_output=False)

            pred_mask = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu().detach(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )

            # assemble the multiple predicted masks for the same image into one
            pred_mask = torch.any(pred_mask[0], dim=0).type(torch.uint8)

        else:
            pred_mask = torch.zeros(size=(1, w, h), dtype=torch.uint8)

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
    parser.add_argument('--dataset', type=str, default='bbd1k')
    parser.add_argument('--model_name', type=str, default='large', choices=['base', 'large', 'huge'])
    parser.add_argument('--dino_model_name', type=str, default='SwinT', choices=['SwinT', 'SwinB'])
    parser.add_argument('--text_prompt', type=str, default='building', help='format for multiple text prompts: "chair . person . dog ."')
    parser.add_argument('--box_threshold', type=float, default=0.35, help='choose the boxes whose highest similarities are higher than this value')
    parser.add_argument('--text_threshold', type=float, default=0.25, help='extract the words whose similarities are higher than this value')
    parser.add_argument('--out_dir', type=str, default='results/')

    args = parser.parse_args()

    main(args)

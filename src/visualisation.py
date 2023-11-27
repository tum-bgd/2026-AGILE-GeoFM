import matplotlib.pyplot as plt
import numpy as np


class Visualizer():
    def __init__(self, prompt_type):
        self.prompt_type = prompt_type

    def show_prompts(self, ax, prompts):
        if self.prompt_type == 'bb':
            for box in prompts:
                x0, y0 = box[0], box[1]
                w, h = box[2] - box[0], box[3] - box[1]
                ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

        else:
            input_points = np.array(input_points).reshape(-1, 2)
            labels = np.ones_like(input_points[:, 0])
            if self.prompt_type == 'foreground_background_pts':
                labels[labels.shape//2:] = 0
            show_points(input_points, labels, ax)

    def save(self, sample, pred_image):
        fig, axes = plt.subplots(1, 3, figsize=(15, 15))

        axes[0].imshow(sample['image'])
        self.show_prompts(axes[0], sample['prompts'])
        axes[0].title.set_text("Orthophoto")
        axes[0].axis("off")

        axes[1].imshow(sample['gt'], cmap='binary')
        axes[1].title.set_text("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(pred_image, cmap='binary')
        axes[2].title.set_text("Predicted")
        axes[2].axis("off")

        plt.savefig(sample['pred_file'])


def show_points(coords, labels, ax):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=100, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=100, edgecolor='white', linewidth=1.25)

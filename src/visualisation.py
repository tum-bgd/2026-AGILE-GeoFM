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
                ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=1))
        else:
            input_points = np.array(input_points).reshape(-1, 2)
            labels = np.ones_like(input_points[:, 0])

            if self.prompt_type == 'foreground_background_pts':
                labels[labels.shape//2:] = 0

            pos_points = input_points[labels==1]
            neg_points = input_points[labels==0]
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='blue', marker='*', s=100, edgecolor='white', linewidth=1.25)
            ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=100, edgecolor='white', linewidth=1.25)

    def save(self, image, prompts, mask, pred_mask, pred_name):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image)
        self.show_prompts(axes[0], prompts)
        axes[0].title.set_text("Orthophoto")

        axes[1].imshow(mask, cmap='binary')
        axes[1].title.set_text("Ground Truth")
        axes[1].set_frame_on(True)

        axes[2].imshow(pred_mask, cmap='binary')
        axes[2].title.set_text("Predicted")

        for ax in axes:
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.setp(ax.spines.values(), color='gray')

        plt.savefig(pred_name, bbox_inches='tight')
        plt.close()

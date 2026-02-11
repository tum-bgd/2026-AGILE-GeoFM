import matplotlib.pyplot as plt
import numpy as np


class Visualizer():
    def __init__(self, prompt_type, prompt_info):
        self.prompt_type = prompt_type
        self.prompt_info = prompt_info
        self.predictions = {
            'bb': "Bounding Box Prediction",
            'center_pt': "Representative Point Prediction",
            'multiple_pts': f"{prompt_info} Sampled Points Prediction",
            'foreground_background_pts': f"{prompt_info} Fore-/Background Points Prediction",
            'text_prompt': "Grounding DINO Prompted SAM",
            'auto_sam_classified': "CLIP Classified SAM Automatic",
            'remote_sam': f"Text-Prompted RemoteSAM",
        }

    def show_prompts(self, ax, prompts):
        if self.prompt_type == 'bb' or self.prompt_type == 'text_prompt':
            for box in prompts:
                x0, y0 = box[0], box[1]
                w, h = box[2] - box[0], box[3] - box[1]
                ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=1))

        elif self.prompt_type == 'auto_sam_classified':
            for mask in prompts:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
                ax.imshow(mask_image)

        elif self.prompt_type == 'remote_sam':
            pass

        else:
            prompts = np.array(prompts).reshape(-1, 2)
            labels = np.ones_like(prompts[:, 0])

            if self.prompt_type == 'foreground_background_pts':
                labels = np.array([1]*self.prompt_info + [0]*self.prompt_info)

            pos_points = prompts[labels==1]
            neg_points = prompts[labels==0]
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='blue', marker='.', s=100, edgecolor='white', linewidth=1)
            ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=100, edgecolor='white', linewidth=1)

    def save(self, image, prompts, mask, pred_mask, pred_name):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image)
        self.show_prompts(axes[0], prompts)
        if self.prompt_type == 'text_prompt':
            axes[0].title.set_text(f"Grounding DINO Predicted Prompts")
        elif self.prompt_type == 'auto_sam_classified':
            axes[0].title.set_text("SAM Automatic Orthophoto")
        else:
            axes[0].title.set_text("Prompted Orthophoto")

        axes[1].imshow(mask, cmap='binary')
        axes[1].title.set_text("Ground Truth")

        axes[2].imshow(pred_mask, cmap='binary')
        axes[2].title.set_text(self.predictions[self.prompt_type])

        for ax in axes:
            ax.set_autoscale_on(False)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.setp(ax.spines.values(), color='gray')
            ax.set_xlim([0, 1024])
            ax.set_ylim([1024, 0])

        plt.savefig(pred_name, bbox_inches='tight')
        plt.close()

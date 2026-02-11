'''
Adapted from the official implementation of the paper
Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification
https://github.com/gaopengcuhk/Tip-Adapter/tree/main
'''

import argparse
import open_clip
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm


class Datum:
    def __init__(self, impath='', label=0):
        self._impath = impath
        self._label = label

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label


class DatasetWrapper(TorchDataset):
    def __init__(self, data_source, input_size, transform=None):
        self.data_source = data_source
        self.transform = transform

        # Build transform that doesn't apply any data augmentation
        interp_mode = T.InterpolationMode.BICUBIC
        to_tensor = []
        to_tensor += [T.Resize(input_size, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        normalize = T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        )
        to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            'label': item.label,
            'impath': item.impath
        }

        img0 = Image.open(item.impath).convert('RGB')

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = 'img'
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output['img'] = img

        return output['img'], output['label']

    def _transform_image(self, tfm, img0):
        img = tfm(img0)
        return img


class BuildDataset:
    def __init__(self, cache_dir, num_shots):
        classnames = ['foreground', 'background']
        train = []
        for i, classname in enumerate(classnames):
            class_path = os.path.join(cache_dir, classname)
            for file in os.listdir(class_path):
                path = os.path.join(class_path, file)
                item = Datum(impath=path, label=i)
                train.append(item)

        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        self._train = train
        self._num_classes = len(classnames)

    @property
    def train(self):
        return self._train

    @property
    def num_classes(self):
        return self._num_classes

    def generate_fewshot_dataset(self, *data_sources, num_shots=-1, repeat=False):
        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output

    def split_dataset_by_label(self, data_source):
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def build_cache_model(cfg, clip_model, train_loader_cache):
    cache_keys = []
    cache_values = []

    with torch.no_grad():
        # Data augmentation for the cache model
        for augment_idx in range(cfg['augment_epoch']):
            train_features = []

            print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
            for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                images = images.cuda()
                image_features = clip_model.encode_image(images)
                train_features.append(image_features)
                if augment_idx == 0:
                    target = target.cuda()
                    cache_values.append(target)
            cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

    cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
    cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
    cache_keys = cache_keys.permute(1, 0)
    cache_values = F.one_hot(torch.cat(cache_values, dim=0)).float()

    torch.save(cache_keys, os.path.join(cfg['cache_dir'], 'keys_' + str(cfg['shots']) + "shots_" + cfg['clip_model_name'] + ".pt"))
    torch.save(cache_values, os.path.join(cfg['cache_dir'], 'values_' + str(cfg['shots']) + "shots_" + cfg['clip_model_name'] + ".pt"))

    return cache_keys, cache_values


def run_tip_adapter_F(cfg, cache_keys, cache_values, clip_weights, clip_model, train_loader_F):
    # Enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(next(clip_model.parameters()).dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))

    beta, alpha = cfg['beta'], cfg['alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            loss = F.cross_entropy(tip_logits, target)

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        if (correct_samples / all_samples) > best_acc:
            best_acc = correct_samples / all_samples
            best_epoch = train_idx
            torch.save(adapter.weight, os.path.join(cfg['cache_dir'], "best_F_" + str(cfg['shots']) + "shots_" + cfg['clip_model_name'] + ".pt"))

    print(f"**** After fine-tuning, Tip-Adapter-F's best train accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")


def main(args):
    clip_model_name = 'ViT-bigG-14'
    label = args.label
    cfg = {
        'clip_model_name': clip_model_name,
        'shots': args.shots,
        'lr': args.lr,
        'augment_epoch': args.augment_epoch,
        'train_epoch': args.train_epoch,
        'beta': args.beta,
        'alpha': args.alpha,
        'cache_dir': args.cache_dir,
    }

    os.makedirs(cfg['cache_dir'], exist_ok=True)

    # Load OpenClip
    clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name, pretrained='laion2b_s39b_b160k')
    tokenizer = open_clip.get_tokenizer(clip_model_name)
    clip_model = clip_model.cuda().eval()

    # Textual features
    text_queries = [f"satellite image of {label}", f"satellite image of background"]
    with torch.no_grad():
        texts = tokenizer(text_queries).cuda()
        text_features = clip_model.encode_text(texts)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    clip_weights = text_features.T

    data = BuildDataset(cfg['cache_dir'], cfg['shots']).train
    train_tranform = T.Compose([
        T.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    train_loader_cache = torch.utils.data.DataLoader(
        DatasetWrapper(data, input_size=224, transform=train_tranform),
        batch_size=256,
        num_workers=8,
        shuffle=False,
        drop_last=False,
        pin_memory=(torch.cuda.is_available())
    )
    assert len(train_loader_cache) > 0

    train_loader_F = torch.utils.data.DataLoader(
        DatasetWrapper(data, input_size=224, transform=train_tranform),
        batch_size=256,
        num_workers=8,
        shuffle=True,
        drop_last=False,
        pin_memory=(torch.cuda.is_available())
    )
    assert len(train_loader_F) > 0

    # Construct the cache model by few-shot training set
    cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)

    run_tip_adapter_F(cfg, cache_keys, cache_values, clip_weights, clip_model, train_loader_F)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=str, default='a surface water')
    parser.add_argument('--shots', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--augment_epoch', type=int, default=10)
    parser.add_argument('--train_epoch', type=int, default=20)
    parser.add_argument('--beta', type=float, default=5)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--cache_dir', type=str, default='data/water1k_cache/')

    args = parser.parse_args()

    main(args)

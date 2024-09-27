import os
import numpy as np
from PIL import Image
import cv2
from pycocotools.coco import COCO
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A
import json
import torch
import torchvision.transforms as T


class DatasetAugs(torch.utils.data.Dataset):
    """
    PyTorch dataset koji radi sa COCO anotacijama i omogućava čitanje i augmentaciju slika, maski, i bounding box-ova za instance segmentation zadatke. 
    
    Args:
        root (str): Putanja do direktorijuma gde se nalaze slike.
        annotation_files (list of str): Lista putanja do COCO JSON fajlova sa anotacijama.
        transforms (list, optional): Lista transformacija koje će se primeniti na slike i maske. Default je None.
        size_factor (int, optional): Faktor za uvećanje skupa podataka, koristi se za oversampling. Default je 1.
        resize_pair (tuple, optional): Par (height, width) za promenu veličine slike i maski. Default je None.
        train (bool, optional): Zastavica da li je dataset za treniranje ili evaluaciju (False za evaluaciju). Default je True.
    """
        
    def __init__(self, root, annotation_files, transforms=None, size_factor = 1, resize_pair = None, train = True):
        self.root = root
        self.transforms = transforms
        self.coco_datasets = [COCO(annotation_file) for annotation_file in annotation_files]
        self.img_ids = []
        for coco in self.coco_datasets:
            self.img_ids.extend(list(coco.imgs.keys()))
        self.img_ids = list(set(self.img_ids))
        self.size_factor = size_factor
        self.resize_pair = resize_pair
        self.train = train

    def __getitem__(self, idx):
        img_id = self.img_ids[idx % len(self.img_ids)]
        for coco in self.coco_datasets:
            if img_id in coco.imgs:
                img_info = coco.loadImgs(img_id)[0]
                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)
                break

        img_path = os.path.join(self.root, img_info['file_name'])
        img = np.array(Image.open(img_path).convert("RGB"))

        boxes = []
        labels = []
        masks = []
        areas = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            mask = coco.annToMask(ann)
            masks.append(mask)
            areas.append(ann['area'])


        image_id = torch.tensor([img_id])
        iscrowd = torch.as_tensor([ann.get('iscrowd', 0) for ann in anns], dtype=torch.int64)
        masks = [np.array(mask) for mask in masks]
        
        bbox_params = {'format':'pascal_voc', 'min_area': 5, 'min_visibility': 0.5, 'label_fields': ['category_id']}
        
        if self.resize_pair is not None:
            resize_transform = A.Compose([A.Resize(self.resize_pair[0], self.resize_pair[1], interpolation=cv2.INTER_LINEAR, p=1)], bbox_params=bbox_params, p=1)
            resized = resize_transform(image=img, masks=masks, bboxes=boxes, category_id=labels)
            img = resized['image']
            masks = resized['masks']
            boxes = resized['bboxes']

        
        if self.transforms is not None and idx > len(self.img_ids):

            augs = A.Compose(self.transforms, bbox_params=bbox_params, p=1)
            augmented = augs(image=img, masks=masks, bboxes=boxes, category_id=labels)
            img = augmented['image']
            masks = augmented['masks']
            boxes = augmented['bboxes']

        #img = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0 # Convert to tensor, channels first
        if self.train:
            img = T.ToTensor()(img)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        return img, target

    def __len__(self):
        return len(self.img_ids) * self.size_factor

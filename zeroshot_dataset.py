import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

DIRECTIONS = ['left', 'right', 'up', 'down']
# SHAPES = ['rectangle', 'circle', 'triangle', 'star', 'ellipse', 
#           'pentagon', 'hexagon', 'diamond', 'cross', 'heart']

SHAPES = ['apple', 'baby', 'banana', 'bear', 
          'car', 'check', 'cherries', 'cigarette', 'clip', 
          'cloud', 'cursor', 'dice', 'fuel', 'hacker', 'instagram', 
          'japan', 'keyboard', 'korea', 'leaf', 'man-walking', 'moon', 
          'motor', 'pencil', 'school-bus', 'school', 'seat', 'thunder', 'tornado', 'tower']

COLORS = ['red', 'orange', 'yellow', 'green', 'blue', 
          'navy', 'purple', 'pink', 'cyan', 'brown']

TYPE_NAMES = ["direction_only", "direction_sentence", "visual_desc", "color_only", "shape_only"]
# TYPE_NAMES_CLS = ["direction_only", "direction_sentence", "color_only", "shape_only", "desc_direction"]
TYPE_NAMES_CLS = ["direction_only", "direction_sentence", "shape_only", "desc_direction_obj"]


def get_class_texts(type_name):
    if type_name == "direction_only":
        return DIRECTIONS
    elif type_name == "direction_sentence":
        return [f"The object is located at the {d}" for d in DIRECTIONS]
    elif type_name == "color_only":
        return COLORS
    elif type_name == "shape_only":
        return SHAPES
    elif type_name == "desc_direction":
        return None
    elif type_name == "desc_direction_obj":
        return None
    else:
        raise ValueError(f"Unknown type: {type_name}")


def get_class_list(type_name):
    if type_name in ["direction_only", "direction_sentence", "desc_direction", "desc_direction_obj"]:
        return DIRECTIONS
    elif type_name == "color_only":
        return COLORS
    elif type_name == "shape_only":
        return SHAPES
    else:
        raise ValueError(f"Unknown type: {type_name}")


def get_desc_direction_texts(color, shape):
    return [f"The {color} {shape} is located at the {d}" for d in DIRECTIONS]

def get_desc_direction_obj_texts(shape):
    return [f"The {shape} is located at the {d}" for d in DIRECTIONS]


class ZeroShotDataset(Dataset):
    def __init__(self, csv_path, data_root, transform, mode="sim"):
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.transform = transform
        self.mode = mode
        
        # 컬럼 존재 여부 미리 체크
        self.has_color = 'color' in self.df.columns
        self.has_shape = 'shape' in self.df.columns
        
        if mode == "cls":
            self.direction_to_idx = {d: i for i, d in enumerate(DIRECTIONS)}
            self.color_to_idx = {c: i for i, c in enumerate(COLORS)}
            self.shape_to_idx = {s: i for i, s in enumerate(SHAPES)}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_path = os.path.join(self.data_root, row['image_path'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        direction = row['direction'].replace('_', ' ')
        color = row['color'] if self.has_color else None
        shape = row['shape'] if self.has_shape else None
        
        if self.mode == "sim":
            gt = {
                "direction_only": direction,
                "direction_sentence": f"The object is located at the {direction}",
            }
            if color and shape:
                gt["visual_desc"] = f"The object in the frame is {color} {shape}"
                gt["color_only"] = color
            if shape:
                gt["shape_only"] = shape
        else:
            gt = {
                "direction_only": self.direction_to_idx.get(direction, -1),
                "direction_sentence": self.direction_to_idx.get(direction, -1),
            }
            if color:
                gt["color_only"] = self.color_to_idx[color]
            if shape:
                gt["shape_only"] = self.shape_to_idx[shape]
            
            # 동적 텍스트
            gt["desc_direction"] = self.direction_to_idx.get(direction, -1)
            gt["desc_direction_obj"] = self.direction_to_idx.get(direction, -1)
            
            if color and shape:
                gt["_desc_texts"] = get_desc_direction_texts(color, shape)
            if shape:
                gt["_desc_texts_obj"] = get_desc_direction_obj_texts(shape)
        
        return image, gt


def collate_fn_sim(batch):
    images = torch.stack([item[0] for item in batch])
    gt_texts = {t: [item[1][t] for item in batch] for t in TYPE_NAMES}
    return images, gt_texts


def collate_fn_cls(batch):
    images = torch.stack([item[0] for item in batch])
    gt_indices = {t: torch.tensor([item[1][t] for item in batch]) for t in TYPE_NAMES_CLS}
    
    dynamic_texts = {}
    if "desc_direction" in TYPE_NAMES_CLS:
        dynamic_texts["desc_direction"] = [item[1]["_desc_texts"] for item in batch]
    if "desc_direction_obj" in TYPE_NAMES_CLS:
        dynamic_texts["desc_direction_obj"] = [item[1]["_desc_texts_obj"] for item in batch]
    
    return images, gt_indices, dynamic_texts
import os
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from torchvision import datasets, transforms


def save_results(test_labels, predictions, out_dir, task_name):
    # Confusion matrix 저장
    save_dir = os.path.join(out_dir, task_name)
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(test_labels, predictions)
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(os.path.join(save_dir,'confusion_matrix.csv'), index=True)

    # Prediction results 저장
    results_df = pd.DataFrame({
        'true_label': test_labels,
        'predicted_label': predictions,
        'correct': test_labels == predictions
    })
    results_df.to_csv(os.path.join(save_dir,'prediction_results.csv'), index=False)

    # (Optional) Classification report도 저장
    report = classification_report(test_labels, predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(save_dir,'classification_report.csv'), index=True)

    return



def data_transform():
    train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.CenterCrop(224),          # geometry는 고정(방향 불변)
    # transforms.ColorJitter(
    #     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
    # ),
    # transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224,224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    return train_tf, val_tf


import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

import decord
from decord import VideoReader, cpu

# decord.bridge.set_bridge('torch')


class VideoFolderDataset(Dataset):
    """ImageFolder 스타일의 Video Dataset"""
    
    def __init__(self, root_dir, processor, num_frames=16, extensions=('.mp4', '.avi', '.mov', '.mkv')):
        self.root_dir = root_dir
        self.processor = processor
        self.num_frames = num_frames
        
        self.classes = sorted([d for d in os.listdir(root_dir) 
                               if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(extensions):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls_name]))
        
        print(f"Found {len(self.samples)} videos in {len(self.classes)} classes")
    
    def _sample_frames(self, video_path):
        """비디오에서 균등 간격으로 프레임 샘플링"""
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()  # ✅ .asnumpy() 명시적 사용
        
        # ✅ (T, H, W, C), uint8 보장
        frames = frames.astype(np.uint8)
        frames = frames.transpose(0, 3, 1, 2)
    
        return frames
    
    def __len__(self):
        return len(self.samples)
    
    # def __getitem__(self, idx):
    #     video_path, label = self.samples[idx]
        
    #     try:
    #         frames = self._sample_frames(video_path)  # (T, H, W, C)
            
    #         # processor 적용 - list of frames 형태로 전달
    #         inputs = self.processor(list(frames), return_tensors="pt")
    #         pixel_values = inputs['pixel_values']  # (1, T, C, H, W)
            
    #         # (1, T, C, H, W) -> (C, T, H, W) for VideoMAE
    #         pixel_values = pixel_values.squeeze(0).permute(1, 0, 2, 3)
            
    #         return pixel_values, label
            
    #     except Exception as e:
    #         print(f"Error loading {video_path}: {e}")
    #         # 에러 시 dummy 반환
    #         return torch.zeros(3, self.num_frames, 224, 224), label
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        try:
            frames = self._sample_frames(video_path)  # (T, H, W, C), numpy array
            
            # ✅ 수정: list 대신 numpy array 그대로 전달
            inputs = self.processor(list(frames), return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)  # (1, T, C, H, W)
            pixel_values = pixel_values.permute(1, 0, 2, 3)
            
            return pixel_values, label
            
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            return torch.zeros(3, self.num_frames, 224, 224), label
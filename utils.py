import os
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from torchvision import datasets, transforms
import numpy as np

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

def save_zeroshot_results(results, type_names, out_dir, task_name, mode="sim"):
    """
    mode: "sim" → similarity 결과 저장 (기존)
          "cls" → classification 결과 저장 (accuracy, confusion matrix 등)
    """
    save_dir = os.path.join(out_dir, task_name)
    os.makedirs(save_dir, exist_ok=True)
    
    if mode == "sim":
        # 1. Raw results (행: 이미지, 열: type)
        raw_df = pd.DataFrame(results)
        raw_df.to_csv(os.path.join(save_dir, 'similarity_raw.csv'), index=False)
        
        # 2. Summary (type별 mean, std)
        summary_data = []
        for type_name in type_names:
            sims = results[type_name]
            summary_data.append({
                'type': type_name,
                'mean': np.mean(sims),
                'std': np.std(sims),
                'min': np.min(sims),
                'max': np.max(sims),
            })
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(save_dir, 'similarity_summary.csv'), index=False)
    
    else:  # cls
        from zeroshot_dataset import get_class_list
        
        summary_data = []
        
        for type_name in type_names:
            preds = results[type_name]["preds"]
            gts = results[type_name]["gts"]
            class_names = get_class_list(type_name)
            
            # Accuracy
            correct = sum(p == g for p, g in zip(preds, gts))
            acc = correct / len(gts) * 100
            
            summary_data.append({
                'type': type_name,
                'accuracy': acc,
                'correct': correct,
                'total': len(gts),
                'num_classes': len(class_names)
            })
            
            # Per-type confusion matrix
            cm = confusion_matrix(gts, preds, labels=range(len(class_names)))
            cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
            cm_df.to_csv(os.path.join(save_dir, f'confusion_matrix_{type_name}.csv'))
            
            # Per-type classification report
            report = classification_report(gts, preds, target_names=class_names, 
                                           output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(os.path.join(save_dir, f'classification_report_{type_name}.csv'))
        
        # Summary
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(save_dir, 'accuracy_summary.csv'), index=False)


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
        
        
        
        
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# def plot_tsne(test_features, test_labels, output_dir, task_name):
#     os.makedirs(os.path.join(output_dir, task_name), exist_ok=True)
#     tsne = TSNE(n_components=2, random_state=42, perplexity=30)
#     features_2d = tsne.fit_transform(test_features)
    
#     # 클래스별로 따로 그리기
#     plt.figure(figsize=(10, 8))
#     unique_labels = np.unique(test_labels)
#     colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
#     for i, label in enumerate(unique_labels):
#         mask = test_labels == label
#         plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
#                    c=[colors[i]], label=f'Class {label}', alpha=0.6, s=10)
    
#     plt.legend()
#     plt.title('t-SNE Visualization')
#     plt.xlabel('t-SNE 1')
#     plt.ylabel('t-SNE 2')
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, task_name, 'tsne_visualization.png'), dpi=150)
#     plt.close()
    
    
# def plot_tsne(test_features, test_labels, output_dir, task_name):
#     os.makedirs(os.path.join(output_dir, task_name), exist_ok=True)
    
#     tsne = TSNE(n_components=2, random_state=42, perplexity=30)
#     features_2d = tsne.fit_transform(test_features)
    
#     unique_labels = np.unique(test_labels)
#     colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
#     # 클래스 그룹 정의
#     groups = {
#         'class_0_3': [0, 3],
#         'class_1_2': [1, 2]
#     }
    
#     for group_name, class_list in groups.items():
#         plt.figure(figsize=(10, 8))
        
#         for label in class_list:
#             mask = test_labels == label
#             color_idx = list(unique_labels).index(label)
#             plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
#                        c=[colors[color_idx]], label=f'Class {label}', alpha=0.6, s=10)
        
#         plt.legend()
#         plt.title(f't-SNE Visualization ({group_name})')
#         plt.xlabel('t-SNE 1')
#         plt.ylabel('t-SNE 2')
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, task_name, f'tsne_{group_name}.png'), dpi=150)
#         plt.close()
        
def plot_tsne(test_features, test_labels, output_dir, task_name, max_samples=1000):
    
    os.makedirs(os.path.join(output_dir, task_name), exist_ok=True)
    # 샘플링 (점 개수 줄이기)
    if len(test_features) > max_samples:
        idx = np.random.choice(len(test_features), max_samples, replace=False)
        plot_features = test_features[idx]
        plot_labels = test_labels[idx]
    else:
        plot_features = test_features
        plot_labels = test_labels
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(plot_features)
    
    # 클래스별 색상 직접 지정 (확실히 구분되게)
    color_map = {
        0: 'tab:blue',
        1: 'tab:orange', 
        2: 'tab:green',
        3: 'tab:red'
    }
    
    groups = {
        'class_0_3': [0, 3],
        'class_1_2': [1, 2]
    }
    
    for group_name, class_list in groups.items():
        plt.figure(figsize=(10, 8))
        
        for label in class_list:
            mask = plot_labels == label
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=color_map[label], label=f'Class {label}', 
                       alpha=0.5, s=15, edgecolors='white', linewidths=0.3)
        
        plt.legend(markerscale=2)
        plt.title(f't-SNE Visualization ({group_name})')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, task_name, f'tsne_{group_name}.png'), dpi=150)
        plt.close()
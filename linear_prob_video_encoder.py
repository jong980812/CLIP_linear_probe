from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig
import numpy as np
import torch
from utils import VideoFolderDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import os
import torch
import argparse
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
# from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import utils
import torch





def get_video_features(loader, model, device):
    """VideoMAEv2용 feature extraction"""
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for pixel_values, labels in tqdm(loader):
            # (B, C, T, H, W)
            pixel_values = pixel_values.to(device)
            
            outputs = model(pixel_values)
            
            all_features.append(outputs.cpu().float())
            all_labels.append(labels)

    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()


def main():
    args = parse_args()
    # ============ Config ============
    train_dir = '/data/dataset/vlm_direction/2D_direction_video_linear_probing/train'  # 클래스별 폴더 구조
    val_dir = '/data/dataset/vlm_direction/2D_direction_video_linear_probing/val'
    cache_dir = '/data/dataset/LLaVA-Video-100K-Subset/'
    batch_size = 16
    num_workers = 8
    num_frames = 16
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ============ Model & Processor ============
    print("Loading VideoMAEv2-Base...")
    config = AutoConfig.from_pretrained(
        "OpenGVLab/VideoMAEv2-Base", 
        trust_remote_code=True, 
        cache_dir=cache_dir
    )
    processor = VideoMAEImageProcessor.from_pretrained(
        "OpenGVLab/VideoMAEv2-Base", 
        cache_dir=cache_dir
    )
    model = AutoModel.from_pretrained(
        'OpenGVLab/VideoMAEv2-Base', 
        config=config, 
        trust_remote_code=True, 
        cache_dir=cache_dir
    ).to(device).eval()
    
    # ============ Dataset & DataLoader ============
    print("Loading datasets...")
    train_dataset = VideoFolderDataset(train_dir, processor, num_frames=num_frames)
    val_dataset = VideoFolderDataset(val_dir, processor, num_frames=num_frames)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # ============ Feature Extraction ============
    print("Extracting train features...")
    train_features, train_labels = get_video_features(train_loader, model, device)
    print(f"Train features shape: {train_features.shape}")
    
    print("Extracting val features...")
    test_features, test_labels = get_video_features(val_loader, model, device)
    print(f"Test features shape: {test_features.shape}")
    
    # ============ Linear Probe ============
    print("Training classifier...")
    classifier = LogisticRegression(
        random_state=0, 
        C=0.316, 
        max_iter=1000, 
        verbose=1
    )
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
    # ============ Evaluation ============
    train_acc = classifier.score(train_features, train_labels)
    test_acc = classifier.score(test_features, test_labels)
    
    print(f"\nTrain Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    utils.save_results(
        test_labels,
        predictions,
        out_dir=args.output_dir,
        task_name=args.task_name
        )
    print(f"Results saved to {args.output_dir}")
    
    with open(os.path.join(args.output_dir, args.task_name, "_acc.txt"), "a") as f:
        f.write(f"Accuracy = {accuracy:.3f}\n")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)


    return parser.parse_args()


if __name__ == "__main__":
    main()
    
    
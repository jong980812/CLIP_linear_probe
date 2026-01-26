import os
import clip
import torch
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
# from torchvision.datasets import CIFAR100
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

"""
실행 방법:
python linear_prob.py --data_root "/local_datasets/object_direction_2D_linear_probing" --output_dir "/data/sunghun/proj/CLIP/results/linear_probe_on_object_direction" --task_name "udlr" --cls_token
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--object_type", type=str, default=None)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="ViT-B/16")
    parser.add_argument(
        "--cls_token",
        action="store_true",
        help="Use CLS token only (otherwise use patch tokens)"
    )

    return parser.parse_args()

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

def get_features(loader, model, device, is_cls=False):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader):
            features = model.encode_image(images.to(device), is_cls=is_cls)

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

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

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Proceeding on {device}")
    model, _ = clip.load(args.model_type, device, download_root="/data/dataset/LLaVA-Video-100K-Subset/clip_weight")

    # datasets
    train_tf, val_tf = data_transform()
    train_dataset = datasets.ImageFolder(f"{args.data_root}/train", transform=train_tf)
    val_dataset   = datasets.ImageFolder(f"{args.data_root}/val",   transform=val_tf)

    # dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1024,
        shuffle=False,   # 중요: val은 shuffle X
        num_workers=8,
        pin_memory=True
    )

    # Calculate the image features
    train_features, train_labels = get_features(train_loader, model, device, is_cls=args.cls_token)
    test_features, test_labels   = get_features(val_loader,   model, device, is_cls=args.cls_token)

    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)
    
    model.eval()
    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(float)) * 100.

    os.makedirs(args.output_dir, exist_ok=True)
    save_results(
        test_labels,
        predictions,
        out_dir=args.output_dir,
        task_name=args.task_name
        )
    print(f"Results saved to {args.output_dir}")
    print(f"Accuracy = {accuracy:.3f}")
    
    with open(os.path.join(args.output_dir, args.task_name, "_acc.txt"), "a") as f:
        f.write(f"Accuracy = {accuracy:.3f}\n")

if __name__ == "__main__":
    main()
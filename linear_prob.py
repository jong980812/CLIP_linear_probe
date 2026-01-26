import os
import clip
from clip.model import CLIPFeatureExtractor
import torch
import argparse
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
# from torchvision.datasets import CIFAR100
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import utils
"""
실행 방법:
python linear_prob.py --data_root "/local_datasets/object_direction_2D_linear_probing" --output_dir "/data/sunghun/proj/CLIP/results/linear_probe_on_object_direction" --task_name "udlr" --cls_token
"""




def get_features(loader, model, device, is_cls=False):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader):
            features = model.encode_image(images.to(device), is_cls=is_cls)
            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# def get_features_from_idx(loader, model, device, is_cls=False, layer_idx=-2):
#     all_features = []
#     all_labels = []
    
#     with torch.no_grad():
#         for images, labels in tqdm(loader):
#             features = model.encode_image(images.to(device), layer_idx=layer_idx, is_cls=is_cls)

#             all_features.append(features)
#             all_labels.append(labels)

#     return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()
def get_features_from_idx(loader, model, device, is_cls=False, layer_idx=-2):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader):
            features = model.encode_image(images.to(device), layer_idx=layer_idx, is_cls=is_cls)

            all_features.append(features.cpu())  # ✅ 바로 CPU로 이동
            all_labels.append(labels)

    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Proceeding on {device}")
    model, preprocess = clip.load(args.model_type, device, download_root="/data/dataset/LLaVA-Video-100K-Subset/clip_weight")
    model.eval()
#!
    extractor = CLIPFeatureExtractor(model)

    # 이미지 준비

    # # 마지막에서 두 번째 layer의 feature 추출
    # feature = extractor.encode_image(image, layer_idx=-2, is_cls=True)
    # print(f"Feature shape: {feature.shape}")  # (4, 1024)
#!
    # datasets
    train_tf, val_tf = utils.data_transform()
    train_dataset = datasets.ImageFolder(f"{args.data_root}/train", transform=train_tf)
    val_dataset   = datasets.ImageFolder(f"{args.data_root}/val",   transform=val_tf)

    # dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,   # 중요: val은 shuffle X
        num_workers=8,
        pin_memory=True
    )

    if args.layer_idx is not None:
        train_features, train_labels = get_features_from_idx(train_loader, extractor, device, is_cls=args.cls_token, layer_idx=args.layer_idx)
        test_features, test_labels   = get_features_from_idx(val_loader,   extractor, device, is_cls=args.cls_token, layer_idx=args.layer_idx)
    # Calculate the image features
    else:
        train_features, train_labels = get_features(train_loader, model, device, is_cls=args.cls_token)
        test_features, test_labels   = get_features(val_loader,   model, device, is_cls=args.cls_token)

    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)
    
    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(float)) * 100.

    os.makedirs(args.output_dir, exist_ok=True)
    utils.save_results(
        test_labels,
        predictions,
        out_dir=args.output_dir,
        task_name=args.task_name
        )
    print(f"Results saved to {args.output_dir}")
    print(f"Accuracy = {accuracy:.3f}")
    
    with open(os.path.join(args.output_dir, args.task_name, "_acc.txt"), "a") as f:
        f.write(f"Accuracy = {accuracy:.3f}\n")



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
    parser.add_argument(
        "--layer_idx",
        type=int,
        default=None,
        help="If specified, extract features from this layer index"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
    
    
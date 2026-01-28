
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
import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
"""
실행 방법:
python linear_prob.py --data_root "/local_datasets/object_direction_2D_linear_probing" --output_dir "/data/sunghun/proj/CLIP/results/linear_probe_on_object_direction" --task_name "udlr" --cls_token
"""






def get_features(loader, model, device):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader):
            features = model(images.to(torch.bfloat16).to(device))

            all_features.append(features['pooler_output'].cpu().float())  # ✅ 바로 CPU로 이동
            all_labels.append(labels)

    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Proceeding on {device}")


    model = AutoModel.from_pretrained(
        'OpenGVLab/InternViT-300M-448px',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        cache_dir='/data/dataset/LLaVA-Video-100K-Subset/'
        ).cuda().eval()

    image_processor = CLIPImageProcessor.from_pretrained('OpenGVLab/InternViT-300M-448px')

    # pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
    # pixel_values = pixel_values.to(torch.bfloat16).cuda()
    train_tf = transforms.Compose([
    transforms.Resize((448,448)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    val_tf = transforms.Compose([
        transforms.Resize((448,448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    # outputs = model(pixel_values)
    model.eval()
    # datasets
    train_tf, val_tf = utils.data_transform()
    train_dataset = datasets.ImageFolder(f"{args.data_root}/train", transform=train_tf)
    val_dataset   = datasets.ImageFolder(f"/data/jongseo/project/CLIP/data/images",   transform=val_tf)

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

    train_features, train_labels = get_features(train_loader, model, device)
    test_features, test_labels   = get_features(val_loader,   model, device)

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
    parser.add_argument("--task_name", type=str, required=True)


    return parser.parse_args()


if __name__ == "__main__":
    main()
    
    
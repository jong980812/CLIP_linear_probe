import os
import clip
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils

from zeroshot_dataset import ZeroShotDataset, collate_fn, TYPE_NAMES
"""
실행 방법:
python zero_shot.py \
    --data_root "/local_datasets/object_position_9_class_linear_probing/val" \
    --csv_path "val.csv" \
    --output_dir "./results/zero_shot" \
    --task_name "position_zeroshot"
"""

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Proceeding on {device}")
    model, preprocess = clip.load(args.model_type, device, download_root="/data/dataset/LLaVA-Video-100K-Subset/clip_weight")
    model.eval()

    dataset = ZeroShotDataset(args.csv_path, args.data_root, preprocess)

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    results = {t: [] for t in TYPE_NAMES}

    with torch.no_grad():
        for images, gt_texts in tqdm(loader):
            images = images.to(device)
            batch_size = images.size(0)
            
            # Encode images: [batch, dim]
            image_features = model.encode_image(images, use_proj=True)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            all_texts = [t for type_name in TYPE_NAMES for t in gt_texts[type_name]]
            text_inputs = clip.tokenize(all_texts).to(device)
            
            text_features = model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # [batch*5, dim] → [batch, 5, dim]
            text_features = text_features.view(len(TYPE_NAMES), batch_size, -1).permute(1, 0, 2)
            
            # [batch, 1, dim] @ [batch, dim, 5] → [batch, 5]
            sims = torch.bmm(
                image_features.unsqueeze(1),
                text_features.permute(0, 2, 1)
            ).squeeze(1)
            
            for i, type_name in enumerate(TYPE_NAMES):
                results[type_name].extend(sims[:, i].cpu().tolist())

    print("\n" + "="*50)
    print("Results (Mean Similarity)")
    print("="*50)
    for type_name in TYPE_NAMES:
        mean_sim = np.mean(results[type_name])
        std_sim = np.std(results[type_name])
        print(f"{type_name:20s}: {mean_sim:.4f} ± {std_sim:.4f}")
    
    utils.save_zeroshot_results(results, TYPE_NAMES, args.output_dir, args.task_name)
    print(f"\nResults saved to {os.path.join(args.output_dir, args.task_name)}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--object_type", type=str, default=None)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="ViT-B/16")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--use_proj", action="store_true", help="Use projection head for feature extraction")

    return parser.parse_args()


if __name__ == "__main__":
    main()
    
    
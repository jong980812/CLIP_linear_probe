import os
import clip
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils

from zeroshot_dataset import (
    ZeroShotDataset, collate_fn_cls, 
    TYPE_NAMES_CLS, get_class_texts, get_class_list
)

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model, preprocess = clip.load(args.model_type, device, 
                                   download_root="/data/dataset/LLaVA-Video-100K-Subset/clip_weight")
    model.eval()

    dataset = ZeroShotDataset(args.csv_path, args.data_root, preprocess, mode="cls")
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn_cls
    )

    print("Encoding text features...")

    # 동적 처리할 type 목록
    DYNAMIC_TYPES = ["desc_direction", "desc_direction_obj"]

    with torch.no_grad():
        text_features = {}
        for t in TYPE_NAMES_CLS:
            if t in DYNAMIC_TYPES:
                continue
            tokens = clip.tokenize(get_class_texts(t)).to(device)
            feat = model.encode_text(tokens)
            text_features[t] = feat / feat.norm(dim=-1, keepdim=True)
            print(f"  {t}: {len(get_class_texts(t))} classes")

    results = {t: {"preds": [], "gts": []} for t in TYPE_NAMES_CLS}

    with torch.no_grad():
        for images, gt_indices, dynamic_texts in tqdm(loader):
            images = images.to(device)
            batch_size = images.size(0)

            image_features = model.encode_image(images, use_proj=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logit_scale = model.logit_scale.exp()
            
            # 고정 text features 처리
            for type_name in TYPE_NAMES_CLS:
                if type_name in DYNAMIC_TYPES:
                    continue
                    
                logits = logit_scale * image_features @ text_features[type_name].T
                preds = logits.argmax(dim=-1)
                
                gt = gt_indices[type_name].to(device)
                valid = gt >= 0
                
                results[type_name]["preds"].extend(preds[valid].cpu().tolist())
                results[type_name]["gts"].extend(gt[valid].cpu().tolist())
            
            # 동적 text 처리 (있는 것만)
            for type_name in DYNAMIC_TYPES:
                if type_name not in TYPE_NAMES_CLS:
                    continue
                
                texts_list = dynamic_texts[type_name]
                all_texts = [t for texts in texts_list for t in texts]
                text_tokens = clip.tokenize(all_texts).to(device)
                text_feat = model.encode_text(text_tokens)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                text_feat = text_feat.view(batch_size, 4, -1)
                
                logits = logit_scale * torch.bmm(
                    image_features.unsqueeze(1), 
                    text_feat.permute(0, 2, 1)
                ).squeeze(1)
                
                preds = logits.argmax(dim=-1)
                gt = gt_indices[type_name].to(device)
                valid = gt >= 0
                
                results[type_name]["preds"].extend(preds[valid].cpu().tolist())
                results[type_name]["gts"].extend(gt[valid].cpu().tolist())

    print("\n" + "="*60)
    print("Zero-Shot Classification Accuracy")
    print("="*60)
    for type_name in TYPE_NAMES_CLS:
        preds = results[type_name]["preds"]
        gts = results[type_name]["gts"]
        correct = sum(p == g for p, g in zip(preds, gts))
        acc = correct / len(gts) * 100
        print(f"{type_name:20s}: {acc:6.2f}% ({correct}/{len(gts)})")
    
    utils.save_zeroshot_results(results, TYPE_NAMES_CLS, args.output_dir, args.task_name, mode="cls")
    print(f"\nResults saved to {os.path.join(args.output_dir, args.task_name)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="ViT-B/16")
    parser.add_argument("--csv_path", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    main()
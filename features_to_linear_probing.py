import torch
import numpy as np
import argparse
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, 
                        help="Path to feature pt files directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results")
    parser.add_argument("--task_name", type=str, default="linear_probe",
                        help="Task name for saving results")
    parser.add_argument("--mode", type=str, choices=["video", "image", "pooled"], required=True,
                        help="Data mode: video or image")
    parser.add_argument("--C", type=float, default=0.316,
                        help="Regularization strength for LogisticRegression")
    parser.add_argument("--max_iter", type=int, default=1000,
                        help="Max iterations for LogisticRegression")
    return parser.parse_args()


def process_features(features, mode):
    if mode == "video":
        # (N, F, 729, 896) → (N, 729, 896) → (N, 896)
        pooled = features.mean(dim=1)  # frame pooling
        del features
        pooled = pooled.mean(dim=1)    # patch pooling
    elif mode == "image":
        # (N, 729, 896) → (N, 896)
        pooled = features.mean(dim=1)
        del features
    elif mode == "pooled":
        # 이미 (N, D)
        pooled = features
    
    return pooled.float().numpy()


def plot_tsne(features, labels, class_to_idx, save_path):
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
    
    plt.figure(figsize=(10, 8))
    for idx in sorted(idx_to_class.keys()):
        mask = labels == idx
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                    c=colors[idx % len(colors)], label=idx_to_class[idx], alpha=0.7, s=50)
    
    plt.legend(loc='best', fontsize=12)
    plt.title("t-SNE of Test Features", fontsize=14)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"t-SNE saved to {save_path}")


def main():
    args = parse_args()
    
    print(f"Mode: {args.mode}")
    
    # train 로드
    train_data = torch.load(f"{args.data_root}/train_features.pt")
    
    # test 또는 val 파일 찾기
    if os.path.exists(f"{args.data_root}/test_features.pt"):
        test_data = torch.load(f"{args.data_root}/test_features.pt")
    elif os.path.exists(f"{args.data_root}/val_features.pt"):
        test_data = torch.load(f"{args.data_root}/val_features.pt")
    else:
        raise FileNotFoundError("No test or val features found")
    
    print(f"Raw train shape: {train_data['features'].shape}")
    print(f"Raw test shape: {test_data['features'].shape}")
    
    train_features = process_features(train_data['features'], args.mode)
    train_labels = train_data['labels'].numpy()
    
    test_features = process_features(test_data['features'], args.mode)
    test_labels = test_data['labels'].numpy()
    
    print(f"Processed - Train: {train_features.shape}, Test: {test_features.shape}")
    
    # types는 optional
    train_types = train_data.get('types', None)
    test_types = test_data.get('types', None)
    if train_types is not None:
        print(f"Train types: {len(train_types) if isinstance(train_types, list) else train_types}")
        print(f"Test types: {len(test_types) if isinstance(test_types, list) else test_types}")
    
    # t-SNE 시각화
    if args.output_dir:
        os.makedirs(os.path.join(args.output_dir, args.task_name), exist_ok=True)
        tsne_path = os.path.join(args.output_dir, args.task_name, "tsne_test.png")
        plot_tsne(test_features, test_labels, train_data['class_to_idx'], tsne_path)
    
    # Logistic Regression
    classifier = LogisticRegression(random_state=0, C=args.C, max_iter=args.max_iter, verbose=1)
    classifier.fit(train_features, train_labels)
    
    # Evaluate
    predictions = classifier.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions) * 100
    
    print(f"\nAccuracy: {accuracy:.2f}%")
    
    idx_to_class = {v: k for k, v in train_data['class_to_idx'].items()}
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions, target_names=target_names))
    
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, predictions))
    
    # 결과 저장
    if args.output_dir:
        result_path = os.path.join(args.output_dir, args.task_name, "results.txt")
        with open(result_path, "w") as f:
            f.write(f"Data root: {args.data_root}\n")
            f.write(f"Mode: {args.mode}\n")
            f.write(f"Train: {train_features.shape}, Test: {test_features.shape}\n")
            if train_types:
                f.write(f"Train types: {len(train_types) if isinstance(train_types, list) else train_types}\n")
                f.write(f"Test types: {test_types}\n")
            f.write(f"\nAccuracy: {accuracy:.2f}%\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(test_labels, predictions, target_names=target_names))
            f.write("\nConfusion Matrix:\n")
            f.write(str(confusion_matrix(test_labels, predictions)))
        
        print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()
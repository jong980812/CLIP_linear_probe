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
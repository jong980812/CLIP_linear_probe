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
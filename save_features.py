import av
import torch
import numpy as np
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from tqdm import tqdm
from glob import glob
import os

def read_video_pyav(video_path, num_frames=4):
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    if total_frames == 0:
        total_frames = sum(1 for _ in container.decode(video=0))
        container.seek(0)
    indices = np.arange(0, total_frames, total_frames / num_frames).astype(int)
    
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i > indices[-1]:
            break
        if i in indices:
            frames.append(frame)
    container.close()
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


# 설정
data_root = "/local_datasets/2D_direction_video_symmetry_4class_1combo"
save_root = "/data2/local_datasets/vlm_features/object_direction_E2E/VideoLLaVA_8frames"
num_frames = 8

os.makedirs(save_root, exist_ok=True)

# 모델 로드
print("Loading model...")
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = VideoLlavaForConditionalGeneration.from_pretrained(
    "LanguageBind/Video-LLaVA-7B-hf", 
    # quantization_config=quantization_config,
    device_map="auto",
    cache_dir="/data/dataset/LLaVA-Video-100K-Subset"
)

processor = VideoLlavaProcessor.from_pretrained(
    "LanguageBind/Video-LLaVA-7B-hf", 
    cache_dir="/data/dataset/LLaVA-Video-100K-Subset"
)
model.eval()
print("Model loaded!")

class_to_idx = {"down": 0, "left": 1, "right": 2, "up": 3}
prompt = "USER: <video>\nDescribe. ASSISTANT:"

for split in ["train", "val"]:
    print(f"\n{'='*50}")
    print(f"Processing {split}")
    
    video_paths = glob(f"{data_root}/{split}/*/*.mp4")
    print(f"Found {len(video_paths)} videos")
    
    all_features = []
    all_labels = []
    error_count = 0
    
    pbar = tqdm(video_paths, desc=split, ncols=100)
    
    with torch.no_grad():
        for idx, video_path in enumerate(pbar):
            label = video_path.split("/")[-2]
            
            try:
                video = read_video_pyav(video_path, num_frames=num_frames)
                inputs = processor(text=prompt, videos=video, return_tensors="pt").to('cuda')
                
                out = model.model.get_video_features(inputs['pixel_values_videos'])[0]

                pooled_feature = out.mean(dim=0)
                del out
                pooled_feature = pooled_feature[1:].mean(dim=0)

                all_features.append(pooled_feature.cpu())
                all_labels.append(class_to_idx[label])
                
                del inputs, pooled_feature
                torch.cuda.empty_cache()
                
                # 진행 상황 업데이트
                pbar.set_postfix({
                    'done': len(all_features),
                    'err': error_count,
                    'label': label
                })
                
            except Exception as e:
                error_count += 1
                pbar.set_postfix({'err': error_count, 'last_err': str(e)[:20]})
                continue
    
    print(f"\nProcessed: {len(all_features)}, Errors: {error_count}")
    
    # 저장
    features = torch.stack(all_features)
    labels = torch.tensor(all_labels)
    
    save_dict = {
        'features': features,
        'labels': labels,
        'class_to_idx': class_to_idx,
        'num_frames': num_frames
    }
    
    save_path = os.path.join(save_root, f"{split}_features.pt")
    torch.save(save_dict, save_path)
    
    print(f"Saved {split}: {features.shape}")
    print(f"Labels distribution: {torch.bincount(labels)}")

print("\n" + "="*50)
print("Done!")
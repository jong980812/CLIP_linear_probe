import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor

model = AutoModel.from_pretrained(
    'OpenGVLab/InternViT-300M-448px',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    cache_dir='/data/dataset/LLaVA-Video-100K-Subset/'
    ).cuda().eval()

image = Image.open('./examples/image1.jpg').convert('RGB')

image_processor = CLIPImageProcessor.from_pretrained('OpenGVLab/InternViT-300M-448px')

pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
pixel_values = pixel_values.to(torch.bfloat16).cuda()

outputs = model(pixel_values)
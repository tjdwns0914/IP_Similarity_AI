import os
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# 1. 모델 로드
print("DINOv2 AI 모델을 로드하는 중입니다...")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
model = AutoModel.from_pretrained('facebook/dinov2-small')

# 2. 이미지 데이터 불러오기 (경로는 데스크탑 세팅 시 맞춰서 수정 예정)
dataset_path = "./CapturedDataset/bugi" 
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
    print(f"임시 폴더 '{dataset_path}'를 생성했습니다. 나중에 여기에 유니티 이미지를 넣으세요.")

image_files = [f for f in os.listdir(dataset_path) if f.endswith('.png')]
print(f"{len(image_files)}장의 데이터를 불러왔습니다.")

# 3. FAISS DB 인덱싱 (이미지가 있을 경우에만 실행)
if len(image_files) > 0:
    embedding_dim = 384 
    index = faiss.IndexFlatL2(embedding_dim) 
    
    model.eval()
    with torch.no_grad():
        for img_name in image_files:
            img_path = os.path.join(dataset_path, img_name)
            image = Image.open(img_path).convert('RGB')
            
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            index.add(np.array([features]))
    print("FAISS DB 구축 완료!")
else:
    print("이미지가 없어 DB 구축을 건너뜁니다. 코드 세팅은 완료되었습니다!")
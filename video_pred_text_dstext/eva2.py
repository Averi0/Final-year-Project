import torch
from model_utils import VideoTextModel, get_tokenizer
from video_utils import extract_frames, extract_img_features
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import os

model = VideoTextModel()
model.load_state_dict(torch.load("saved_model/video_caption_model.pt"))
model.eval()

tokenizer = get_tokenizer()
samples = json.load(open("dstext_dataset/annotations.json"))

y_true = []
y_pred = []

for item in samples:
    video_path = os.path.join("dstext_dataset", item["video"])

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        continue

    try:
        frames = extract_frames(video_path)
        if not frames:
            raise ValueError("No frames extracted.")
        img_feat = extract_img_features(frames).unsqueeze(0)
    except Exception as e:
        print(f"Skipping video {video_path}: {e}")
        continue

    tokens = tokenizer(item["caption"], return_tensors="pt", padding='max_length', max_length=20, truncation=True)

    with torch.no_grad():
        output = model(img_feat, tokens["input_ids"], tokens["attention_mask"])

    pred_ids = torch.argmax(output, dim=-1).tolist()
    true_ids = tokens["input_ids"].squeeze()[0].item()  # Using CLS token

    y_pred.append(pred_ids[0])
    y_true.append(true_ids)

# Calculate metrics
print("Precision:", precision_score(y_true, y_pred, average='macro'))
print("Recall:", recall_score(y_true, y_pred, average='macro'))
print("F1 Score:", f1_score(y_true, y_pred, average='macro'))

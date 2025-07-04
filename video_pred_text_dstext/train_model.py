import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model_utils import VideoTextModel, get_tokenizer
from video_utils import extract_frames, extract_img_features
import os
import json

class DSTextDataset(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.tokenizer = get_tokenizer()
        self.samples = json.load(open(os.path.join(data_folder, 'annotations.json')))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        video_path = os.path.join(self.data_folder, item["video"])
        caption = item["caption"]

        try:
            frames = extract_img_features(extract_frames(video_path))
        except Exception as e:
            print(f"Skipping video {video_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        tokens = self.tokenizer(caption, return_tensors="pt", padding='max_length', max_length=20, truncation=True)
        return frames, tokens["input_ids"].squeeze(), tokens["attention_mask"].squeeze()

model = VideoTextModel()
criterion = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_dataset = DSTextDataset("dstext_dataset")
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

for epoch in range(10):
    print(f"Epoch {epoch+1}/5")
    model.train()
    for img_feat, input_ids, att_mask in train_loader:
        optimizer.zero_grad()
        output = model(img_feat, input_ids, att_mask)  # [B, seq_len, vocab]
        loss = criterion(output.view(-1, output.size(-1)), input_ids.view(-1))
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item():.4f}")

os.makedirs("saved_model", exist_ok=True)
torch.save(model.state_dict(), "saved_model/video_caption_model.pt")

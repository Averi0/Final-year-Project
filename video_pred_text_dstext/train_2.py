import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model_utils import VideoTextModel, get_tokenizer
from video_utils import extract_frames, extract_img_features
import os
import json


# Custom Dataset class for DSText
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
            return self.__getitem__((idx + 1) % len(self))  # Retry next item

        tokens = self.tokenizer(caption, return_tensors="pt", padding='max_length', max_length=20, truncation=True)

        # Use only the first token (CLS token) as label
        label = tokens["input_ids"].squeeze()[0]  # 0D scalar (int)

        return frames, tokens["input_ids"].squeeze(), tokens["attention_mask"].squeeze(), label


# Initialize model, optimizer, loss
model = VideoTextModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Load dataset
train_dataset = DSTextDataset("dstext_dataset")
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Training loop
for epoch in range(5):
    print(f"Epoch {epoch + 1}/5")
    model.train()
    for img_feat, input_ids, att_mask, labels in train_loader:
        optimizer.zero_grad()
        output = model(img_feat, input_ids, att_mask)  # output shape: [B, vocab_dim]
        loss = criterion(output, labels)  # labels: [B] scalar token IDs
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item():.4f}")

# Save trained model
os.makedirs("saved_model", exist_ok=True)
torch.save(model.state_dict(), "saved_model/video_caption_model.pt")

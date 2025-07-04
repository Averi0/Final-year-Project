import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class VideoTextModel(nn.Module):
    def __init__(self, text_dim=768, img_dim=2048, hidden_dim=512, output_dim=768):
        super(VideoTextModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.img_fc = nn.Linear(img_dim, hidden_dim)
        self.text_fc = nn.Linear(text_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, img_feat, input_ids, attention_mask):
        text_feat = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        img_proj = self.img_fc(img_feat)
        text_proj = self.text_fc(text_feat)
        combined = img_proj + text_proj
        output = self.classifier(combined)
        return output

def get_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")

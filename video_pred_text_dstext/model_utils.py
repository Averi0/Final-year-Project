import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

def get_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")

class VideoTextModel(nn.Module):
    def __init__(self):
        super(VideoTextModel, self).__init__()
        self.tokenizer = get_tokenizer()
        self.vocab_size = self.tokenizer.vocab_size

        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.image_proj = nn.Linear(2048, 768)  # project image feat to BERT hidden dim
        self.decoder = nn.GRU(768, 768, batch_first=True)
        self.fc = nn.Linear(768, self.vocab_size)

    def forward(self, img_feat, input_ids, attention_mask):
        """
        img_feat: [B, 2048]
        input_ids: [B, seq_len]
        attention_mask: [B, seq_len]
        """
        B, seq_len = input_ids.shape
        img_proj = self.image_proj(img_feat).unsqueeze(1)  # [B, 1, 768]

        token_embeds = self.bert.embeddings(input_ids)  # [B, seq_len, 768]
        decoder_input = torch.cat((img_proj, token_embeds[:, :-1, :]), dim=1)  # shift right

        output, _ = self.decoder(decoder_input)  # [B, seq_len, 768]
        logits = self.fc(output)  # [B, seq_len, vocab_size]
        return logits

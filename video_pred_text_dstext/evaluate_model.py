import torch
from model_utils import VideoTextModel, get_tokenizer
from video_utils import extract_frames, extract_img_features
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
import os

import nltk
nltk.download('punkt')


# Load model and tokenizer
model = VideoTextModel()
model.load_state_dict(torch.load("saved_model/video_caption_model.pt", map_location="cpu"))
model.eval()

tokenizer = get_tokenizer()
samples = json.load(open("dstext_dataset/annotations.json"))
smooth_fn = SmoothingFunction().method4

bleu_scores = []
exact_matches = 0
total_samples = 0

for item in samples:
    video_path = os.path.join("dstext_dataset", item["video"])
    reference_caption = item["caption"]

    if not os.path.exists(video_path):
        print(f"Skipping missing video: {video_path}")
        continue

    try:
        frames = extract_frames(video_path)
        if not frames:
            continue
        img_feat = extract_img_features(frames).unsqueeze(0)

        # Greedy decoding
        generated_ids = [tokenizer.cls_token_id]
        for _ in range(20):
            input_ids = torch.tensor([generated_ids])
            attn_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                output = model(img_feat, input_ids, attn_mask)
            next_token = torch.argmax(output[0, -1]).item()
            if next_token in [tokenizer.sep_token_id, tokenizer.pad_token_id]:
                break
            generated_ids.append(next_token)

        pred_caption = tokenizer.decode(generated_ids[1:], skip_special_tokens=True).strip()
        ref_caption = reference_caption.strip()

        # Token-level BLEU
        pred_tokens = pred_caption.split()
        ref_tokens = ref_caption.split()
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_fn)
        bleu_scores.append(bleu)

        # Exact match
        if pred_caption.lower() == ref_caption.lower():
            exact_matches += 1

        total_samples += 1

    except Exception as e:
        print(f"Error processing {video_path}: {e}")

# Final metrics
avg_bleu = sum(bleu_scores) / total_samples
exact_match_acc = exact_matches / total_samples

print(f"Total Samples: {total_samples}")
print(f"Average BLEU Score: {avg_bleu:.4f}")
print(f"Exact Match Accuracy: {exact_match_acc:.2%}")

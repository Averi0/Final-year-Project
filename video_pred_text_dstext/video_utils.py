import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import resnet50

# Load ResNet50 and remove the classification head
resnet = resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

# Define preprocessing pipeline
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def extract_frames(video_path, max_frames=3):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1
    cap.release()
    return frames


def extract_img_features(frames):
    if not frames:
        raise ValueError("No frames extracted from video. Check video path or format.")

    with torch.no_grad():
        feats = []
        for frame in frames:
            try:
                img = transform(frame).unsqueeze(0)
                feat = resnet(img).squeeze().detach()
                feats.append(feat)
            except Exception as e:
                print(f"Frame processing error: {e}")

        if not feats:
            raise ValueError("All frames failed during feature extraction.")

        return torch.stack(feats).mean(dim=0)

import streamlit as st
import torch
from model_utils import VideoTextModel, get_tokenizer
from video_utils import extract_frames, extract_img_features
import os

st.set_page_config(page_title="🎥 Video_to_Text", layout="centered")
st.title("🎬 Video_to_Text Prediction App")

@st.cache_resource
def load_model():
    model = VideoTextModel()
    model.load_state_dict(torch.load("saved_model/video_caption_model.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()
tokenizer = get_tokenizer()

video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if video_file:
    temp_path = "temp_uploaded_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(video_file.read())

    st.video(temp_path)

    try:
        frames = extract_frames(temp_path)
        if not frames:
            st.error("❌ Could not extract frames.")
        else:
            img_feat = extract_img_features(frames).unsqueeze(0)
            generated = [tokenizer.cls_token_id]

            for _ in range(20):
                input_ids = torch.tensor([generated])
                attn_mask = torch.ones_like(input_ids)
                with torch.no_grad():
                    output = model(img_feat, input_ids, attn_mask)
                next_token = torch.argmax(output[0, -1]).item()
                if next_token == tokenizer.sep_token_id:
                    break
                generated.append(next_token)

            predicted_caption = tokenizer.decode(generated[1:], skip_special_tokens=True)
            st.subheader("📝 Predicted Caption:")
            st.success(predicted_caption)

    except Exception as e:
        st.error(f"Error: {e}")

    if os.path.exists(temp_path):
        os.remove(temp_path)

import streamlit as st
import torch
from model_utils import VideoTextModel, get_tokenizer
from video_utils import extract_frames, extract_img_features
import os

st.set_page_config(page_title="🎥 Video Captioning", layout="centered")
st.title("🎬 Video Captioning with BERT + ResNet50")

# Load model
@st.cache_resource
def load_model():
    model = VideoTextModel()
    model.load_state_dict(torch.load("saved_model/video_caption_model.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()
tokenizer = get_tokenizer()

# Upload interface
video_file = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mov"])

if video_file:
    # Save video temporarily
    temp_path = "temp_uploaded_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(video_file.read())

    st.video(temp_path)

    try:
        # Extract features from video
        frames = extract_frames(temp_path)
        if not frames:
            st.error("⚠️ Could not extract frames. Try another video.")
        else:
            img_feat = extract_img_features(frames).unsqueeze(0)

            # Create dummy text input (needed by model)
            dummy = tokenizer("video", return_tensors="pt", padding='max_length', max_length=20, truncation=True)

            with torch.no_grad():
                output = model(img_feat, dummy['input_ids'], dummy['attention_mask'])

            # Predict single token ID
            predicted_ids = torch.argmax(output, dim=-1).tolist()  # [token_id]
            predicted_caption = tokenizer.decode(predicted_ids, skip_special_tokens=True)

            # Display result
            st.subheader("📝 Predicted Caption:")
            if predicted_caption.strip():
                st.success(f"**{predicted_caption}**")
            else:
                st.warning("⚠️ Unable to decode token. Might be a special/unknown token.")
                st.info(f"Raw Token ID: `{predicted_ids[0]}`")

    except Exception as e:
        st.error(f"❌ Error: {e}")

    # Optional: delete temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

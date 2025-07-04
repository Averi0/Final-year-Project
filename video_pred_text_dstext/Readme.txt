
# Text Prediction from Video Frames

This project presents a lightweight proof-of-concept AI system that predicts text from video frames using a hybrid architecture of computer vision and natural language processing. It integrates ResNet50 for visual feature extraction and BERT for text generation, all wrapped in an interactive Streamlit web interface.

## 🧠 Overview

The system was built to demonstrate the feasibility of video-to-text prediction in a resource-constrained environment (no GPU, low CPU, limited internet). Despite its simplified structure, the prototype effectively showcases multimodal learning between image and language domains.

## ⚙️ Technologies Used

- **Python** – Core programming language
- **PyTorch / Torchvision** – For implementing ResNet50
- **Hugging Face Transformers** – For BERT model integration
- **OpenCV** – For video frame extraction and preprocessing
- **Streamlit** – For building the user interface
- **PyCharm** – IDE used for development

## 🚀 How to Run

1. Clone the repository or copy the project files.
2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. Install required dependencies:
    ```bash
    pip install -r requirements.txt
   
    ```
4.## 🔗 Pretrained Model Download

The model file is too large for GitHub, so it's hosted on Google Drive.

👉 [Download video_caption_model.pt from Google Drive](https://drive.google.com/file/d/1MsoPW5qOC9jA4AW7VcUYrBFNim2unP7j/view?usp=drive_link)

After downloading, place it inside the `saved_model/` directory before running the project.

5. Run the Streamlit app:
    ```bash
   streamlit run streamlit_app.py
    ```

The application will launch in your default browser. Upload a short video clip and see frame-by-frame text predictions.

## 🧪 Evaluation Metrics

| Metric         | Score   |
|----------------|---------|
| BLEU Score     | 0.0000  |
| Token Accuracy | 0.5000  |

- BLEU score suggests weak sentence fluency and structure.
- Token accuracy shows partial vocabulary and object detection alignment.

## 📉 Limitations

- Very small, custom dataset used
- No decoder module for full sentence generation
- Limited training due to lack of GPU and slow internet
- BLEU may not reflect partial semantic correctness

## 💡 Future Work

- Incorporate a decoder module for better sentence-level output
- Train on large-scale datasets (e.g., DSText, LSVTD)
- Use advanced evaluation metrics (e.g., METEOR, ROUGE)
- Optimize for deployment on mobile or edge devices

## 👨‍💻 Author

Developed by [Benstowe Alex Johnson] as part of a final year undergraduate research project (2025).

---

This prototype affirms the potential of combining computer vision and NLP models in resource-constrained environments and lays the foundation for further development and real-world application.

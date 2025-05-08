import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import io
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GOOGLE_GEMINI_API_KEY'))

# Define the blood cell classes
classes = ["RBC", "WBC"]

# Function to load the YOLO model
def load_model(model_path):
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None
    return model

# Perform detection and return results and cell counts
def detect_and_plot(image, model):
    results = model.predict(image)[0]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image[..., ::-1])

    cell_counts = {"RBC": 0, "WBC": 0}

    for detection in results.boxes:
        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
        conf = detection.conf[0].cpu().numpy()
        cls = detection.cls[0].cpu().numpy()

        if conf >= 0.5:
            cell_type = classes[int(cls)]
            cell_counts[cell_type] += 1

            if cell_type == 'WBC':
                edge_color = '#8A2BE2'  # Dark Violet for WBC
                text_bg_color = '#8A2BE2'
            else:
                edge_color = '#006400'  # Dark Green for RBC
                text_bg_color = '#006400'

            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=edge_color, facecolor='none'
            )
            ax.add_patch(rect)

            plt.text(
                x1, y1, f"{cell_type} {conf:.2f}",
                color='white', fontsize=10, backgroundcolor=text_bg_color
            )

    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    return buf, cell_counts

# Get Gemini info about WBC or RBC
def get_gemini_response(cell_type):
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""Give a short medical overview about {cell_type} (White Blood Cell or Red Blood Cell). Include:
1. Main function in the body
2. Typical normal range
3. Common conditions if levels are too high or too low

Format using markdown with headings and bullet points."""
    response = model.generate_content(prompt)
    return response.text.strip()

# Handle user custom questions about blood cells
def get_gemini_response_for_query(user_query):
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""You are a helpful assistant focused on hematology, human blood analysis, and cell biology. 
Respond only to questions related to:

- Blood components like RBC & WBC
- Blood disorders or infections
- Functions and significance of different blood cells

If a question is unrelated to human blood or biology, respond with:

"I'm sorry, I cannot answer questions not related to human blood or cell biology."

**User's question:** {user_query}

Use markdown with headings and bullet points to structure your answer."""
    response = model.generate_content(prompt)
    return response.text.strip()

# Streamlit UI setup
st.set_page_config(page_title="Red Blood Cell (RBC) & White Blood Cell (WBC), Detection, Classification & Counting System in Blood Smear Image", layout="centered")
st.markdown("<h1 style='text-align: center; color: #FF0800;'>Red Blood Cell (RBC) & White Blood Cell (WBC), Detection, Classification & Counting System in Blood Smear Image</h1>", unsafe_allow_html=True)

# Initialize session state
if 'chat_visible' not in st.session_state:
    st.session_state.chat_visible = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Image upload section
st.subheader("Upload Blood Smear Image:")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open and process the image
    image = Image.open(uploaded_image).convert("RGB")
    image = image.resize((256, 256))  # Resize to 256x256

    st.subheader("Uploaded Blood Smear Image:")
    st.image(image, caption='Uploaded Image (Resized to 256x256)', use_container_width=True)

    image_np_bgr = np.array(image)[..., ::-1] 

    model_path = "BLOOD_RBC_WBC_Detector_model.pt"  # Update to your model path
    model = load_model(model_path)

    if model is not None:
        result_plot, cell_counts = detect_and_plot(image_np_bgr, model)

        st.subheader("Detection Results:")
        st.image(result_plot, caption='Detection Results', use_container_width=True)

        total_cells = sum(cell_counts.values())
        if total_cells == 0:
            st.subheader("No WBC or RBC detected with sufficient confidence.")
        else:
            st.subheader("Detected Blood Cells Count:")
            st.markdown(f"- **White Blood Cells (WBC):** {cell_counts['WBC']}")
            st.markdown(f"- **Red Blood Cells (RBC):** {cell_counts['RBC']}")

            st.markdown("---")
            st.subheader("Cell Type Information:")
            for cell_type, count in cell_counts.items():
                if count > 0:
                    st.markdown(f"### {cell_type}")
                    cell_info = get_gemini_response(cell_type)
                    st.markdown(cell_info)
                    st.markdown("---")

# Chat assistant section
if st.session_state.chat_visible:
    st.title("Blood Cell Chat Assistant")
    st.write("Ask questions about blood cell biology!")

    for user_input, bot_response in st.session_state.chat_history:
        st.markdown(f"**You:** {user_input}")
        st.markdown(f"**Bot:** {bot_response}")
        st.markdown("---")

    user_input = st.text_input("Enter your blood-related question:", key="chat_input")

    if user_input:
        bot_response = get_gemini_response_for_query(user_input)
        st.session_state.chat_history.append((user_input, bot_response))
        st.markdown(f"**You:** {user_input}")
        st.markdown(f"**Bot:** {bot_response}")
        st.markdown("---")

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.chat_visible = False

else:
    if st.button("Start Chat Assistant"):
        st.session_state.chat_visible = True

st.empty()

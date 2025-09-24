import streamlit as st
from PIL import Image
import torch
import os

# --------------------------
# LOGIN SECTION
# --------------------------
def login():
    st.title("🔐 Login - Breed Recognition")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "breed" and password == "25004":
            st.session_state["logged_in"] = True
            st.success("✅ Login successful!")
        else:
            st.error("❌ Incorrect username or password.")

# Initialize login state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Show login page
if not st.session_state["logged_in"]:
    login()
    st.stop()

# --------------------------
# MODEL LOADING
# --------------------------
@st.cache_resource
def load_model():
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", trust_repo=True)
    return model

model = load_model()

# --------------------------
# MAIN APP UI
# --------------------------
st.title("🐄 Breed Recognition of Cattles and Buffaloes")
st.markdown("Upload an image and select dataset to detect cattle or buffalo breeds.")

# Dataset option
dataset_options = ["Select Dataset", "Cattle Breeds", "Buffalo Breeds", "Mixed"]
selected_dataset = st.selectbox("📂 Choose Dataset", dataset_options)

# File uploader
uploaded_file = st.file_uploader("📁 Upload an Image...", type=["jpg", "jpeg", "png", "mp4"])

if selected_dataset == "Select Dataset":
    st.warning("⚠️ Please select a dataset to continue.")
    st.stop()

if uploaded_file:
    if uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Run inference
        results = model(image, size=640)

        st.subheader("📌 Prediction Results:")
        st.write(results.pandas().xyxy[0])  # Show prediction table

        # Save and display output image
        output_dir = "runs/detect"
        results.save(save_dir=output_dir)

        # Find the most recent folder created
        latest_folder = sorted(os.listdir(output_dir))[-1]
        output_image_path = os.path.join(output_dir, latest_folder, uploaded_file.name)

        if os.path.exists(output_image_path):
            st.image(output_image_path, caption="🔍 Detected Image", use_column_width=True)
        else:
            st.warning("⚠️ Unable to find the processed image.")

    elif uploaded_file.type == "video/mp4":
        st.video(uploaded_file)
        st.warning("⚠️ Video prediction is not enabled in this demo. Please upload an image.")


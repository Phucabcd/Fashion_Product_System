import streamlit as st
import numpy as np

from PIL import Image
from predict import predict, CLASS_NAMES
# from recommend import find_similar, preprocess_input

st.set_page_config(page_title="Fashion-MNIST Classifier", layout="centered")
st.title("👕 Fashion-MNIST Classifier")
st.write("Upload một ảnh trang phục (grayscale, hoặc màu cũng được) để model dự đoán.")

# --- Sidebar: chọn model ---
model_type = st.segmented_control(
    "Chọn model",
    options=["cnn", "baseline"],
    format_func=lambda x: "CNN" if x == "cnn" else "Baseline",
    default="cnn"
)

# --- Upload ảnh ---
uploaded_file = st.file_uploader("Chọn ảnh", type=["png", "jpg", "jpeg"])


if uploaded_file is not None:
    # Đọc ảnh, chuyển sang grayscale
    image = Image.open(uploaded_file).convert("L")
    image_array = np.array(image)

    # img_for_similarity = preprocess_input(image_array)
    
    col1, col2 = st.columns([4, 8])

    with col1:
        st.image(image, caption="Ảnh gốc", width=100)

    # --- Predict ---
    label, proba = predict(image_array, model_type=model_type)

    with col2:
        st.subheader("Kết quả dự đoán")
        st.metric(label="Nhãn", value=label)
        st.metric(label="Độ tin cậy", value=f"{proba.max() * 100:.1f}%")

    # --- Biểu đồ xác suất từng lớp ---
    st.subheader("Xác suất theo từng lớp")

    proba_dict = {name: float(p) for name, p in zip(CLASS_NAMES, proba)}
    st.bar_chart(proba_dict)
    
    # st.subheader("Sản phẩm tương tự")
    
    # similar_items = find_similar(img_for_similarity, top_k=5)

    # cols = st.columns(5)
    # for col, item in zip(cols, similar_items):
    #     with col:
    #         st.image(item["image"], caption=f"{item['label']}\n{item['similarity']:.2f}")

else:
    st.info("Vui lòng upload 1 ảnh để bắt đầu dự đoán.")
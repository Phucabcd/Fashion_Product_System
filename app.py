import streamlit as st

from recommender import build_model, build_neighbors, get_recommendations, load_artifacts, load_styles, get_product_info

st.set_page_config(page_title="Fashion Recommendation System", layout="wide")
st.header("Fashion Recommendation System")

image_features, filenames = load_artifacts()
model = build_model()
neighbors = build_neighbors(image_features)
styles_df = load_styles()

upload_file = st.file_uploader("Upload Image")

if upload_file is not None:
    saved_path, recommended_images = get_recommendations(upload_file, model, neighbors, filenames)

    st.subheader("Uploaded Image")
    st.image(saved_path)

    st.subheader("Recommended Images")
    cols = st.columns(5)
    for col, image_path in zip(cols, recommended_images):
        with col:
            st.image(image_path)
            info = get_product_info(image_path, styles_df)
            if info:
                st.markdown(f"**{info['name']}**")
                st.caption(f"{info['category']} · {info['color']}")

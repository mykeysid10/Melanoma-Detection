import streamlit as st
from pipeline import * 
from PIL import Image

model = MelanomaDetection()

def styled_subheader(text, color, font_size):
    return f'<h3 style="color: {color}; font-size: {font_size}rem;">{text}</h3>'

def main():
    # Set css
    st.markdown(
        """
        <style>
        .title {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Set app title
    st.markdown("<h1 class='title'>Melanoma Detection<br><br></h1>", unsafe_allow_html=True)

    # Display sample images
    #st.subheader("Sample Images")
    st.markdown(styled_subheader("Sample Images", "red", 1.6), unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image(r"Dataset\test\benign\ISIC_0052212.jpg", caption="Benign")
    with col2:
        st.image(r"Dataset\test\malignant\ISIC_0015256.jpg", caption="Malignant")

    # Upload file container
    #st.subheader("Upload Image")
    st.markdown(styled_subheader("Upload Image", "black", 1.6), unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    # Perform melanoma detection and display results
    if uploaded_file is not None:
        predicted_json, predicted_image = model.perform_melanoma_detection(uploaded_file)
        #st.subheader("Result")
        st.markdown(styled_subheader("Result", "red", 1.6), unsafe_allow_html=True)
        # Display json output, actual and predicted images
        st.json(predicted_json)
        col1, col2 = st.columns(2)
        with col1:
            #st.subheader("Actual Image")
            # st.markdown("<h6 class='title'>Actual Image</h6>", unsafe_allow_html=True)
            st.image(Image.open(uploaded_file), caption = "Actual Image", width = 320)
        with col2:
            # st.markdown("<h6 class='title'>Predicted Image</h6>", unsafe_allow_html=True)
            st.image(predicted_image, caption = "Predicted Image", width = 320)


if __name__ == "__main__":
    main()

# streamlit run app.py

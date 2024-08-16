# Import required libraries
import base64
import plotly.graph_objects as go
import streamlit as st
import PIL
from ultralytics import YOLO

# Replace the relative path to your weight file
model_path = '/content/best.pt'
model_path11= "/content/runs/detect/train/weights/best.pt"
# Setting page layout
st.set_page_config(
    page_title="RAYS Detection",  # Setting page title
    page_icon="/content/i.jpeg",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",    # Expanding sidebar by default
    
)
def set_background(image_file):

    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)
set_background('d.jpeg')


# Creating main page heading
st.title("detection rays medical images")
st.caption('Updload a photo with this :blue[hand signals]: :+1:')
st.caption('Then click the :blue[Detect Objects] button and check the result.')
# Creating two columns on the main page
col1, col2 = st.columns(2)



# Select box
mbti = st.selectbox(
    'Choose the type of medical radiation',
    ('MRI', 'X_Rays', 'No options'), 
    index=2
)



# Creating sidebar
with st.sidebar:
    st.header("Image Config")     # Adding header to sidebar
    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader(
        "Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    # Model Options
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 40)) / 100




if mbti == 'MRI':
    st.write('Brain MRI tumor detection')


    # Adding image to the first column if image is uploaded
    with col1:
        if source_img:
            # Opening the uploaded image
            uploaded_image = PIL.Image.open(source_img)
            # Adding the uploaded image to the page with a caption
            st.image(source_img,
                    caption="Uploaded Image",
                    use_column_width=True
                    )

    try:
        model = YOLO(model_path)
    except Exception as ex:
        st.error(
            f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    if st.sidebar.button('Detect Objects'):
        res = model.predict(uploaded_image,
                            conf=confidence
                            )
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        with col2:
            st.image(res_plotted,
                    caption='Detected Image',
                    use_column_width=True
                    )
            try:
                with st.expander("Detection Results"):
                    for box in boxes:
                        st.write(box.xywh)
            except Exception as ex:
                st.write("No image is uploaded yet!")

elif mbti == 'X_Rays':
    st.write('You are :green[Activist]')
    # Adding image to the first column if image is uploaded
    with col1:
        if source_img:
            # Opening the uploaded image
            uploaded_image = PIL.Image.open(source_img)
            # Adding the uploaded image to the page with a caption
            st.image(source_img,
                    caption="Uploaded Image",
                    use_column_width=True
                    )

    try:
        model = YOLO(model_path11)
    except Exception as ex:
        st.error(
            f"Unable to load model. Check the specified path: {model_path11}")
        st.error(ex)

    if st.sidebar.button('Detect Objects'):
        res = model.predict(uploaded_image,
                            conf=confidence
                            )
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        with col2:
            st.image(res_plotted,
                    caption='Detected Image',
                    use_column_width=True
                    )
            try:
                with st.expander("Detection Results"):
                    for box in boxes:
                        st.write(box.xywh)
            except Exception as ex:
                st.write("No image is uploaded yet!")

 

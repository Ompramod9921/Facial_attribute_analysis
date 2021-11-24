import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title='Deepface',page_icon='👽')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Facial Attribute Analysis")
st.write("Made with ❤️ by om pramod")
st.markdown("*****")

image_file = st.file_uploader("upload your selfie",type=["png","jpg","jpeg"])

if st.button("Analyze image"):
    try:
        st.markdown("****")
        st.image(image_file,use_column_width=True)
        image_loaded = Image.open(image_file)
        new_image = np.array(image_loaded.convert('RGB')) #converting image into array
        img = cv2.cvtColor(new_image,1) #converting the image from 3 channel image (RGB) into 1 channel image.if you don't convert the image into one channel, open-cv does it automatically.
        prediction = DeepFace.analyze(img_path = img, actions = ['age', 'gender', 'race', 'emotion'])
        st.warning("Analysis summary")
        st.success("Your face emotion is : "+ prediction['dominant_emotion'])
        st.success("Gender recognized as : "+prediction['gender'])
        st.success("Your age is : " +str(prediction['age']))
        st.success("It looks like you belong to "+prediction['dominant_race']+"race")
    except :
        st.error("Face could not be detected. Please confirm that the picture is a face photo")      



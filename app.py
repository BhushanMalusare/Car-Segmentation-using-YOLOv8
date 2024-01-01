import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import streamlit as st
import cv2
import os

#path to yolo trained model
model_path = "runs/segment/train/weights/last.pt"

st.title("Car image Segmentation using YOLOv8")
image = cv2.imread("car_image.jpg")
st.image(image,width=300) 



def image_input():
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image")

        # Save the image to a local file
        uploaded_image_file_path = "APP Data/read images/saved_image.jpg"
        cv2.imwrite(uploaded_image_file_path, image)
        st.success("Predicting results")
        col1, col2 = st.columns(2)
        predictions(model_path,uploaded_image_file_path,col1)
        binary_mask(model_path,uploaded_image_file_path,col2)


def predictions(model_path,uploaded_image_file_path,col1):
    segmented_image_path = "APP Data/predcition images/results.jpg"
    # Load a pretrained YOLOv8n model
    model = YOLO(model_path)

    # Run inference on 'saved_image.jpg'
    results = model(uploaded_image_file_path)  # results list

    # Show the results
    for r in results:
        im_array = r.plot(boxes=False)  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image  
        im.save(segmented_image_path)  # save image

    # Load the image from a local file
    image = cv2.imread(segmented_image_path)

    # Check if the image was loaded successfully
    with col1:
        if image is not None:
            st.subheader("Segmented image", anchor=None, help=None, divider=False)
            st.image(image)
            # Add a download button
            image_file = open(segmented_image_path, "rb")  # Open image in binary read mode
            downloaded_data = image_file.read()
            image_file.close()
            st.download_button(label="Download Image", data=downloaded_data, file_name="downloaded_image.jpg", mime="image/jpeg")
            # deleting data from local machine
            os.remove(segmented_image_path)
        else:
            st.error("Error loading image. Please check the file path and try again.")

def binary_mask(model_path,uploaded_image_file_path,col2):
    binary_segmented_image_path = "APP Data/binary mask images/output.png"
    #reading image for prediction
    img = cv2.imread(uploaded_image_file_path)
    H, W, _ = img.shape
    #feeding image to model for segmentation
    model = YOLO(model_path)
    results = model(img)

    #accessing the results and displaying in the form of image
    for result in results:
        for j, mask in enumerate(result.masks.data):
            mask = mask.numpy() * 255
            mask = cv2.resize(mask, (W, H))
            cv2.imwrite(binary_segmented_image_path, mask)

    image = cv2.imread(binary_segmented_image_path)
    with col2:
        if image is not None:
            st.subheader("Binary Segmented image")
            st.image(image)
            # Add a download button
            image_file = open(binary_segmented_image_path, "rb")  # Open image in binary read mode
            downloaded_data = image_file.read()
            image_file.close()
            st.download_button(label="Download Image", data=downloaded_data, file_name="masked_downloaded_image.jpg", mime="image/png")
            # deleting data from local machine
            os.remove(binary_segmented_image_path)
            os.remove(uploaded_image_file_path)
        else:
            st.error("Error loading image. Please check the file path and try again.")

# Example usage:
image_input()
import cv2
import validators
from PIL import Image
from io import BytesIO

from utils.image_utils import *

captured_image = None
mediapipe_confidence_threshold = 0.5
cap = cv2.VideoCapture(0)


def call_pipeline(detection_model, image, container, image_crop_mode=''):
    if detection_model.lower() == 'mediapipe':
        mediapipe_detection(image, mediapipe_confidence_threshold, container, image_crop_mode)
    elif detection_model.lower() == 'opencv':
        opencv_detection(image, container, image_crop_mode)
    elif detection_model.lower() == 'mtcnn':
        deepface_detection(image, container, image_crop_mode, 'mtcnn')


def image_page():
    global mediapipe_confidence_threshold
    global captured_image

    get_image_model()

    image_crop_mode = st.sidebar.radio(
        'Mode', ('Full Image', 'Cropped Image'), horizontal=True)

    detection_model = st.sidebar.radio(
        'Detection Type', ['Mediapipe', 'OpenCV', 'Mtcnn'], horizontal=True)

    st.sidebar.divider()

    if detection_model == 'Mediapipe':
        mediapipe_confidence_threshold = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.5)

    file_tab, url_tab, webcam_tab = st.tabs(['File', 'URL', 'Webcam'])

    # file tab content
    with file_tab:
        st.write('Upload an image')
        input_image_file = st.file_uploader('Upload Image', type=[
            'png', 'jpg', 'jpeg'], label_visibility='collapsed')

        if input_image_file is not None:
            img_bytes = np.asarray(bytearray(input_image_file.read()), dtype=np.uint8)
            image = cv2.imdecode(img_bytes, 1)
        else:
            image = cv2.imread('./assets/image/multi face.jpg')

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, channels='BGR')
            st.caption('Original Image')

        with col2:
            call_pipeline(detection_model, image, col2, image_crop_mode)

    # url tab content
    with url_tab:
        st.write('Enter the URL of the image')
        url = st.text_input(
            'URL', label_visibility='collapsed', placeholder='Image url')

        if url != "" and not validators.url(url):
            st.toast('Please enter a valid URL', icon='😑')
            st.stop()

        if url != "" and validators.url(url):
            import requests
            response = requests.get(url)
            url_image = Image.open(BytesIO(response.content))
        else:
            url_image = None

        col1, col2 = st.columns(2)
        with col1:
            st.image(
                url_image if url_image is not None else
                'https://cdn.pixabay.com/photo/2015/04/23/22/00/tree-736885_1280.jpg')
            st.caption('Original Image')

        with col2:
            if url_image is not None:
                call_pipeline(detection_model, url_image, col2, image_crop_mode)
                st.caption('New Image')

    # webcam tab content
    with webcam_tab:
        st.write('Webcam')

        col1, col2 = st.columns(2)

        with col1:
            cam_input_image = st.image('./assets/image/cam_thumbnail.jpeg') if captured_image is None \
                else st.image('./assets/image/captured_image.jpg')
            btn1, btn2, btn3 = st.columns(3)

        with col2:
            if captured_image is not None:
                call_pipeline(detection_model, captured_image, col2, image_crop_mode)

        if btn1.button('Start', use_container_width=True):
            start_cam(cam_input_image)

        if btn2.button('Capture', use_container_width=True, type='primary'):
            capture(cam_input_image)
            captured_image = cv2.imread('./assets/image/captured_image.jpg')
            call_pipeline(detection_model, captured_image, col2, image_crop_mode)

        if btn3.button('Clear', use_container_width=True):
            clear_cam()

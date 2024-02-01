import cv2
import streamlit as st

from utils.image_utils import mediapipe_detection, get_image_model
from .image import call_pipeline

mediapipe_confidence_threshold = 0.5


def video_page():
    global mediapipe_confidence_threshold

    get_image_model()

    live = st.sidebar.toggle('Use Webcam', False)
    detection_model = st.sidebar.radio(
        'Detection Type', ['Mediapipe', 'Mtcnn'], horizontal=True)

    st.sidebar.divider()

    if detection_model == 'Mediapipe':
        mediapipe_confidence_threshold = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.5)

    col1, col2 = st.columns([3, 1])
    stream = col1.image('./assets/image/multi face.jpg', use_column_width=True)

    if live:
        video = cv2.VideoCapture(0)
    else:
        video = None

    while live:
        ret, frame = video.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            print("Ignoring empty camera frame.")
            video.release()
            break

        # mediapipe_detection(frame, mediapipe_confidence_threshold, stream)
        call_pipeline(detection_model, frame, stream)

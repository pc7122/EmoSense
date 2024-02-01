import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
import tensorflow as tf
from deepface import DeepFace
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates


# load the cascade file
face_cascade = cv2.CascadeClassifier('./utils/haarcascade_frontalface_alt.xml')

# load mediapipe model and drawing utils
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

model = tf.keras.models.load_model('./model/image/base_1_overfit.h5')
image_classes = np.array(["Angry", "Disgusted", "Fear", "Happy", "Neutral", "Sad", "Surprised"])

need_rescale = False


def start_cam(cam_input_image):
    global cap
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.imwrite('./assets/image/captured_image.jpg', frame)
        cam_input_image.image(frame, channels='BGR')


def capture(cam_input_image):
    clear_cam()
    cam_input_image.image('./assets/image/captured_image.jpg')


def clear_cam():
    try:
        cap.release()
    except NameError:
        st.toast('Start webcam first!', icon='😑')
        st.stop()


def get_image_model():
    global model

    with st.sidebar:
        model_name: str = st.selectbox('Models', (
            'Base 1 with VGG',
            'CNN LSTM 1',
        ), placeholder='Choose image model', index=0)

        check_rescale(model_name)

        if model_name == 'Base 1 with VGG':
            model = tf.keras.models.load_model('./model/image/base_1_overfit.h5')
        elif model_name == 'CNN LSTM 1':
            model = tf.keras.models.load_model('./model/image/lstm_1_emotion.keras')
        else:
            pass


def check_rescale(model_name: str):
    global need_rescale

    if model_name.find('Base') == -1:
        need_rescale = True
    else:
        need_rescale = False


def predict_emotion(image) -> (float, str):
    image = cv2.resize(image, (48, 48))
    image = np.expand_dims(image, axis=0)
    image = image / 255. if need_rescale else image

    y_pred = model.predict(image)
    output = np.argmax(y_pred, axis=1)
    confidence = np.max(y_pred)

    return round(confidence*100, 2), image_classes[output[0]]


def predict_emotion_batch(image):
    print(image.shape)
    y_pred = model.predict(image)
    output = np.argmax(y_pred, axis=1)
    confidence = np.max(y_pred, axis=1)

    return confidence, output


def predict_and_draw_text(image, all_faces, text_coordinate):
    try:
        confidence, output = predict_emotion_batch(np.array(all_faces))

        for idx, coordinate in zip(output, text_coordinate):
            cv2.putText(image, image_classes[idx], coordinate, cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 0), 2, cv2.LINE_AA)
    except Exception:
        print('error')


def predict_and_grid(all_faces, container):
    confidence, output = predict_emotion_batch(np.array(all_faces))

    col1, col2, col3 = container.columns(3)
    for idx, (face, _class) in enumerate(zip(all_faces, output)):
        face = cv2.resize(face, (255, 255))

        grid = col1 if idx % 3 == 0 else col2 if idx % 3 == 1 else col3
        grid.image(face, channels='BGR', use_column_width=True)
        grid.caption(image_classes[_class])


def mediapipe_detection(image, detection_confidence, container=None, mode=''):
    with mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=detection_confidence) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    image_rows, image_cols, _ = image.shape
    image_copy = image.copy()

    if results.detections:
        all_faces = []
        text_coordinate = []

        for detection in results.detections:
            try:
                # draw detection box around the face
                mp_drawing.draw_detection(image_copy, detection)

                # get face coordinates
                box = detection.location_data.relative_bounding_box
                x = _normalized_to_pixel_coordinates(box.xmin, box.ymin, image_cols, image_rows)
                y = _normalized_to_pixel_coordinates(box.xmin + box.width, box.ymin + box.height, image_cols, image_rows)

                # process the image for prediction
                cropped_image = image[x[1] - 10:y[1] + 10, x[0] - 10:y[0] + 10]
                cropped_image = cv2.resize(cropped_image, (48, 48))
                cropped_image = cropped_image / 255. if need_rescale else cropped_image

                all_faces.append(cropped_image)
                text_coordinate.append((x[0], x[1]-20))

                # confidence, emotion = predict_emotion(cropped_image)
                # print(confidence, emotion)
                #
                # cv2.putText(image_copy, emotion, (x[0], x[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #             (255, 255, 0), 2, cv2.LINE_AA)

            except Exception:
                pass

        if mode.lower() == 'cropped image':
            predict_and_grid(all_faces, container)
            return

        predict_and_draw_text(image_copy, all_faces, text_coordinate)
        container.image(image_copy, channels='BGR', use_column_width=True)

        if mode.lower() == 'full image':
            container.caption('New Image')


def opencv_detection(image, container=None, mode=None):
    image_copy = image.copy()

    # detect faces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))

    all_faces = []
    text_coordinate = []

    for (x, y, w, h) in faces:
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cropped_image = image[y:y + h, x:x + w]
        cropped_image = cv2.resize(cropped_image, (48, 48))
        cropped_image = cropped_image / 255. if need_rescale else cropped_image

        all_faces.append(cropped_image)
        text_coordinate.append((x, y))

        # confidence, emotion = predict_emotion(cropped_image)
        # print(confidence, emotion)
        #
        # cv2.putText(image_copy, emotion, (x[0], x[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (255, 255, 0), 2, cv2.LINE_AA)

    if mode.lower() == 'full image':
        predict_and_draw_text(image_copy, all_faces, text_coordinate)
        container.image(image_copy, channels='BGR', use_column_width=True)
        container.caption('New Image')
    else:
        predict_and_grid(all_faces, container)


def deepface_detection(image, container=None, mode=None, detection_model=None):
    if mode.lower() == 'cropped image':
        col1, col2, col3 = container.columns(3)

    image_copy = image.copy()
    results = analyze_emotion(image, detection_model)

    # all_faces = []
    # text_coordinate = []

    for index, result in enumerate(results):
        emotions = result['dominant_emotion']
        region = result['region']

        cv2.rectangle(image_copy, (region['x'], region['y']), (region['x']+region['w'], region['y']+region['h']), (255, 255, 255), 2)

        cropped_image = image[region['y']:region['y'] + region['h'], region['x']:region['x'] + region['w']]
        print(cropped_image.shape)

        # cropped_image = cv2.resize(cropped_image, (48, 48))
        # cropped_image = cropped_image / 255. if need_rescale else cropped_image

        # all_faces.append(cropped_image)
        # text_coordinate.append((region['x'], region['y']-20))

        confidence, emotion = predict_emotion(cropped_image)
        print(confidence, emotion)

        if emotion.lower() != emotions.lower():
            emotion = compare_emotions(emotion, confidence, emotions, result['emotion'][emotions])

        if mode.lower() == 'cropped image':
            cropped_image = cv2.resize(cropped_image, (255, 350))

            grid = col1 if index % 3 == 0 else col2 if index % 3 == 1 else col3
            grid.image(cropped_image, channels='BGR', use_column_width=True)
            grid.caption(emotion)

        cv2.putText(image_copy, emotion, (region['x'], region['y']-20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0), 2, cv2.LINE_AA)

    # image_copy = predict_and_draw_text(image_copy, all_faces, text_coordinate)
    if mode.lower() == 'full image':
        container.image(image_copy, channels='BGR', use_column_width=True)
        container.caption('New Image')
        return

    if mode == '':
        container.image(image_copy, channels='BGR', use_column_width=True)


def analyze_emotion(image, detection_model):
    results = DeepFace.analyze(image, actions=['emotion'], detector_backend=detection_model)
    return results


def compare_emotions(emotion1, confidence1, emotion2, confidence2):
    if confidence1 > confidence2:
        return emotion1
    else:
        return emotion2

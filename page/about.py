import streamlit as st

# about tab information
about_text = open('./assets/About.md', 'r').read()


def about_page():
    about_tab, image_tab, audio_tab = st.tabs(
        ['About', 'Image Model', 'Audio Model'])

    # about tab content
    with about_tab:
        st.title('EmoSense - Emotion Recognition App')
        st.markdown(about_text)
        st.link_button('Github', 'https://github.com/pc7122/EmoSense')

    # image tab content
    with image_tab:
        st.subheader('Image Model Information')

        col1, col2 = st.columns(2)
        with col1:
            st.write('Image Model Architecture')
            st.image(
                './assets/image/model/image_cnn_model.png',
                use_column_width=True
            )

        with col2:
            st.write('Image Model Summary')
            st.markdown(
                '''
                | Layer (type) | Output Shape | Param | Activation |
                | :--- | :--- | :--- | :--- |
                | Input | (None, 48, 48, 3) | 0 | - |
                | VGG 16 | (None, 1, 1, 512) | 14714688 | ReLU |
                | Global Average Pooling | (None, 512) | 0 | - |
                | Dense | (None, 1024) | 525312 | ReLU |
                | Dropout | (None, 1024) | 0 | - |
                | Dense | (None, 1024) | 1049600 | ReLU |
                | Dropout | (None, 1024) | 0 | - |
                | Dense | (None, 7) | 7175 | Softmax |
                '''
            )

    # audio tab content
    with audio_tab:
        st.subheader('Audio Model Information')

        with st.container():
            st.write('Audio Model Architecture')
            st.image(
                './assets/image/model/audio_cnn_lstm_model.drawio.png',
                use_column_width=True
            )

        col1, col2 = st.columns(2)
        with col1:
            st.write('Audio Pipeline')
            st.image(
                './assets/image/model/audio_pipeline.png',
                use_column_width=True
            )

        with col2:
            st.write('Audio Model Summary')
            st.markdown(
                '''
                | Layer (type) | Output Shape | Param | Activation |
                | :--- | :--- | :--- | :--- |
                | Input | (None, 90, 130, 1) | 0 | - |
                | Conv2D | (None, 88, 128, 32) | 320 | ReLU |
                | MaxPooling2D | (None, 44, 64, 32) | 0 | - |
                | BatchNormalization | (None, 44, 64, 32) | 128 | - |
                | Dropout 25% | (None, 44, 64, 32) | 0 | - |
                | Conv2D | (None, 42, 62, 64) | 18496 | ReLU |
                | MaxPooling2D | (None, 21, 31, 64) | 0 | - |
                | BatchNormalization | (None, 21, 31, 64) | 256 | - |
                | Dropout 25% | (None, 21, 31, 64) | 0 | - |
                | TimeDistributed | (None, 21, 1984) | 0 | - |
                | LSTM | (None, 21, 128) | 1081856 | ReLU |
                | LSTM | (None, 64) | 49408 | ReLU |
                | Dense | (None, 64) | 4160 | ReLU |
                | Dropout 50% | (None, 64) | 0 | - |
                | Dense | (None, 8) | 520 | Softmax |
                '''
            )

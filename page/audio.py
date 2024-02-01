from utils.audio_utils import *


def audio_page():
    get_audio_model()

    file_tab, sample_tab, microphone_tab = st.tabs(
        ['File', 'Sample', 'Microphone'])

    # file tab content
    with file_tab:
        st.write('Upload an audio file')
        input_audio_file = st.file_uploader(
            'Upload Audio', type=['wav'], label_visibility='collapsed')

        col1, col2 = st.columns(2)

        if input_audio_file is not None:
            col1.audio(input_audio_file, format='audio/wav')
        else:
            pass

    # sample tab content
    with sample_tab:
        col1, col2 = st.columns(2)

        with col1:
            dataset = st.selectbox('Dataset', ('RAVDESS', 'SAVEE', 'CREMA'), index=None)

        with col2:
            audio_file = st.selectbox('Audio', get_audio_list(dataset), index=None, placeholder='Choose audio file')

        if audio_file is not None:
            st.divider()
            audio_pipeline(f'./data/audio/{audio_file}', original_emotion=True)

    # microphone tab content
    with microphone_tab:
        col1, col2 = st.columns(2)

        if col1.button("Start Recording", use_container_width=True):
            record_and_save(col2, './assets/recorded_audio.wav', duration=4)
            st.divider()
            audio_pipeline('./assets/recorded_audio.wav')



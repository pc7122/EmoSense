import streamlit as st
from streamlit_option_menu import option_menu

from page.about import about_page
from page.image import image_page
from page.audio import audio_page
from page.video import video_page
from assets.style import option_menu_style


# basic streamlit settings and styling
st.set_page_config(layout='wide', page_title='EmoSense')

# load the custom css file
with open('./assets/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# sidebar menu
with st.sidebar:
    st.title('EmoSense')
    st.divider()

    # display the option menu to select the app mode
    app_mode = option_menu(
        menu_title=None,
        options=['About', 'Image', 'Audio', 'Video'],
        icons=['person-fill', 'image-fill', 'mic-fill', 'camera-video-fill'],
        styles=option_menu_style
    )

# list of pages to be displayed in the app mode
page_map = {
    'About': about_page,
    'Image': image_page,
    'Audio': audio_page,
    'Video': video_page,
}

# display the selected page
if app_mode in page_map.keys():
    page_map[app_mode]()

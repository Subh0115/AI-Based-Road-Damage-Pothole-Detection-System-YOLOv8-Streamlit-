import streamlit as st

st.set_page_config(
    page_title="Road Damage Detection Suite",
    page_icon="🛣️",
)

st.title("Road Damage Detection Suite")

st.markdown(
    """
    Introducing a complete suite of Road Damage Detection tools, powered by the YOLOv8 deep learning model trained on the Crowdsensing-based Road Damage Detection Challenge 2022 dataset.
    
    This application is designed to enhance road safety and infrastructure maintenance by swiftly identifying and categorizing various forms of road damage, such as potholes and cracks.

    The model detects four types of damage:
    - Longitudinal Crack
    - Transverse Crack
    - Alligator Crack
    - Potholes

    The model is trained on YOLOv8x using the India CRDDC2022 dataset.

    Use the sidebar to explore the apps: realtime webcam, video, images, and the analytics dashboard. All detections are logged to `data/detections.csv` for later analysis.

   
"""
)
# #### Documentations and Links
# - Github
# Project
# Page[Github](https: // github.com / oracl4 / RoadDamageDetection)
# - You
# can
# reach
# me
# on
# it.mahdi.yusuf @ gmail.com
#
# #### License and Citations
# - Road
# Damage
# Dataset
# from Crowdsensing
#
# -based
# Road
# Damage
# Detection
# Challenge(CRDDC2022)
# - All
# rights
# reserved
# on
# YOLOv8
# license
# permits
# by[Ultralytics](https: // github.com / ultralytics / ultralytics) and [Streamlit](https: // streamlit.io /) framework
st.divider()

st.markdown(
    """
    Built for field readiness: works offline after model download, 1GB upload support, and map-based analytics for operations teams (e.g., NHAI).
    """
)


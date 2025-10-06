import os
import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import streamlit as st

# Deep learning framework
from ultralytics import YOLO

from sample_utils.download import download_file
from sample_utils.logger import append_detection_log


st.set_page_config(
	page_title="Video Detection",
	page_icon="📷",
	layout="centered",
	initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"  # noqa: E501
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
CUSTOM_MODEL_PATH = ROOT / "./models/custom/best.pt"
if not CUSTOM_MODEL_PATH.exists():
	# ensure base model is present for first run
	download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

# Session-specific caching
cache_key = "yolov8_active_model"
if cache_key in st.session_state:
	net = st.session_state[cache_key]
else:
	model_path = CUSTOM_MODEL_PATH if CUSTOM_MODEL_PATH.exists() else MODEL_LOCAL_PATH
	net = YOLO(model_path)
	st.session_state[cache_key] = net

CLASSES = [
	"Longitudinal Crack",
	"Transverse Crack",
	"Alligator Crack",
	"Potholes"
]


# Create temporary folder if doesn't exists
if not os.path.exists('./temp'):
	os.makedirs('./temp')

temp_file_input = "./temp/video_input.mp4"
temp_file_infer = "./temp/video_infer.mp4"


def write_bytesio_to_file(filename, bytesio):
	with open(filename, "wb") as outfile:
		outfile.write(bytesio.getbuffer())


def processVideo(video_file):
	# Write the file into disk
	write_bytesio_to_file(temp_file_input, video_file)

	videoCapture = cv2.VideoCapture(temp_file_input)
	if (videoCapture.isOpened() == False):
		st.error('Error opening the video file')
		return

	_width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
	_height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	_fps = videoCapture.get(cv2.CAP_PROP_FPS)
	_frame_count = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))

	st.write("Width, Height and FPS :", _width, _height, _fps)

	inferenceBarText = "Performing inference on video, please wait."
	inferenceBar = st.progress(0, text=inferenceBarText)

	imageLocation = st.empty()

	fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
	cv2writer = cv2.VideoWriter(temp_file_infer, fourcc_mp4, _fps, (_width, _height))

	_frame_counter = 0
	while (videoCapture.isOpened()):
		ret, frame = videoCapture.read()
		if ret is not True:
			inferenceBar.empty()
			break

		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		_image = np.array(frame)
		image_resized = cv2.resize(_image, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
		results = net.predict(
			image_resized,
			conf=conf_thres,
			iou=iou_thres,
			imgsz=imgsz,
			agnostic_nms=True,
			augment=enable_tta,
			max_det=300,
		)

		for result in results:
			boxes = result.boxes.cpu().numpy()
			for _box in boxes:
				label = CLASSES[int(_box.cls)]
				conf = float(_box.conf)
				x1, y1, x2, y2 = _box.xyxy[0].astype(int)
				append_detection_log(
					source="video",
					class_label=label,
					confidence=conf,
					x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
					latitude=None, longitude=None,
					media_name="uploaded_video.mp4",
				)

		annotated_frame = results[0].plot()
		_image_pred = cv2.resize(annotated_frame, (_width, _height), interpolation=cv2.INTER_AREA)

		_out_frame = cv2.cvtColor(_image_pred, cv2.COLOR_RGB2BGR)
		cv2writer.write(_out_frame)

		imageLocation.image(_image_pred)

		_frame_counter = _frame_counter + 1
		inferenceBar.progress(_frame_counter / max(1, _frame_count), text=inferenceBarText)

	videoCapture.release()
	cv2writer.release()

	st.success("Video Processed!")

	col1, col2 = st.columns(2)
	with col1:
		with open(temp_file_infer, "rb") as f:
			st.download_button(
				label="Download Prediction Video",
				data=f,
				file_name="RDD_Prediction.mp4",
				mime="video/mp4",
				use_container_width=True
			)

	with col2:
		if st.button('Restart Apps', use_container_width=True, type="primary"):
			st.rerun()


st.title("Road Damage Detection - Video")
st.write(
	"Detect the road damage in using Video input. Upload the video and start detecting. This section can be useful for examining and process the recorded videos.")

video_file = st.file_uploader("Upload Video", type=".mp4")
st.caption("There is 1GB limit for video size with .mp4 extension. Resize or cut your video if its bigger than 1GB.")

colA, colB, colC = st.columns(3)
with colA:
	conf_thres = st.slider("Confidence", min_value=0.0, max_value=1.0, value=0.02, step=0.05)
with colB:
	iou_thres = st.slider("IOU (NMS)", min_value=0.1, max_value=0.95, value=0.7, step=0.05)
with colC:
	imgsz = st.select_slider("Image size", options=[640, 960, 1280], value=640)
enable_tta = st.checkbox("Enable TTA", value=False)
st.caption("Tip: Lower confidence/IOU for higher recall; increase size for small potholes.")

if video_file is not None:
	if st.button('Process Video', use_container_width=True, type="secondary"):
		st.warning(f"Processing Video {video_file.name}")
		processVideo(video_file)



import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import streamlit as st

# Deep learning framework
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

from sample_utils.download import download_file
from sample_utils.logger import append_detection_log


st.set_page_config(
	page_title="Image Detection",
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
# Load the model
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


class Detection(NamedTuple):
	class_id: int
	label: str
	score: float
	box: np.ndarray


st.title("Road Damage Detection - Image")
st.write("Detect the road damage in using an Image input. Upload the image and start detecting. This section can be useful for examining baseline data.")

image_file = st.file_uploader("Upload Image", type=['png', 'jpg'])

colA, colB, colC = st.columns(3)
with colA:
	conf_thres = st.slider("Confidence", min_value=0.0, max_value=1.0, value=0.02, step=0.05)
with colB:
	iou_thres = st.slider("IOU (NMS)", min_value=0.1, max_value=0.95, value=0.7, step=0.05)
with colC:
	imgsz = st.select_slider("Image size", options=[640, 960, 1280], value=640)
enable_tta = st.checkbox("Enable TTA", value=False)
st.caption("Tip: Lower confidence and IOU to improve recall; increase image size for small potholes.")

if image_file is not None:
	# Load the image
	image = Image.open(image_file)

	col1, col2 = st.columns(2)

	# Perform inference
	_image = np.array(image)
	h_ori = _image.shape[0]
	w_ori = _image.shape[1]

	image_resized = cv2.resize(_image, (imgsz, imgsz), interpolation = cv2.INTER_AREA)
	results = net.predict(
		image_resized,
		conf=conf_thres,
		iou=iou_thres,
		imgsz=imgsz,
		agnostic_nms=True,
		augment=enable_tta,
		max_det=300,
	)

	# Save the results to log
	for result in results:
		boxes = result.boxes.cpu().numpy()
		for _box in boxes:
			label = CLASSES[int(_box.cls)]
			conf = float(_box.conf)
			x1, y1, x2, y2 = _box.xyxy[0].astype(int)
			append_detection_log(
				source="image",
				class_label=label,
				confidence=conf,
				x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
				latitude=None, longitude=None,
				media_name=image_file.name,
			)

	annotated_frame = results[0].plot()
	_image_pred = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation = cv2.INTER_AREA)

	# Original Image
	with col1:
		st.write("#### Image")
		st.image(_image)

	# Predicted Image
	with col2:
		st.write("#### Predictions")
		st.image(_image_pred)

		# Download predicted image
		buffer = BytesIO()
		_downloadImages = Image.fromarray(_image_pred)
		_downloadImages.save(buffer, format="PNG")
		_downloadImagesByte = buffer.getvalue()

		downloadButton = st.download_button(
			label="Download Prediction Image",
			data=_downloadImagesByte,
			file_name="RDD_Prediction.png",
			mime="image/png"
		)



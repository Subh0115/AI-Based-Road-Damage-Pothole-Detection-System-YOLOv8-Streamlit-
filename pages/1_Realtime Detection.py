import logging
import queue
from pathlib import Path
from typing import List, NamedTuple

import av
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# Deep learning framework
from ultralytics import YOLO

from sample_utils.download import download_file
from sample_utils.get_STUNServer import getSTUNServer
from sample_utils.logger import append_detection_log


# Page config is set once in Home.py

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
	# Prefer custom model if available
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


st.title("Road Damage Detection - Realtime")

st.write(
	"Detect the road damage in realtime using USB Webcam. This can be useful for on-site monitoring with personel on the ground. Select the video input device and start the inference."
)

# JavaScript to update geolocation every 2 seconds
geolocation_js = """
    <script>
    function updateLocation() {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(showPosition, showError);
        } else { 
            console.log("Geolocation is not supported by this browser.");
        }
    }

    function showPosition(position) {
        const latitude = position.coords.latitude;
        const longitude = position.coords.longitude;
        window.localStorage.setItem("latitude", latitude.toFixed(6));
        window.localStorage.setItem("longitude", longitude.toFixed(6));
    }

    function showError(error) {
        // suppress alerts to avoid blocking UX
        console.warn("Geolocation error", error);
    }

    setInterval(updateLocation, 2000);
    updateLocation();
    </script>
"""

st.markdown("### Real-Time Geolocation")
latitude_placeholder = st.empty()
longitude_placeholder = st.empty()

components.html(
	f"""
	{geolocation_js}
	<div>
		<p>Latitude: <span id=\"geo-latitude\"></span></p>
		<p>Longitude: <span id=\"geo-longitude\"></span></p>
	</div>
	""",
	height=100,
)

result_queue: "queue.Queue[List[Detection]]" = queue.Queue()


def _to_float_or_none(value):
	try:
		return float(value)
	except Exception:
		return None


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
	# Get geolocation from session_state if set by client JS via another mechanism
	latitude = _to_float_or_none(st.session_state.get("latitude"))
	longitude = _to_float_or_none(st.session_state.get("longitude"))

	# Read dynamic inference settings from session state so slider changes take effect live
	conf_thres = float(st.session_state.get("rt_conf", 0.02))
	iou_thres = float(st.session_state.get("rt_iou", 0.7))
	imgsz = int(st.session_state.get("rt_imgsz", 640))
	enable_tta = bool(st.session_state.get("rt_tta", False))

	latitude_placeholder.write(f"Latitude: {latitude if latitude is not None else 'N/A'}")
	longitude_placeholder.write(f"Longitude: {longitude if longitude is not None else 'N/A'}")

	image = frame.to_ndarray(format="bgr24")
	h_ori = image.shape[0]
	w_ori = image.shape[1]
	# Throttle detection to keep preview responsive and show raw video immediately
	st.session_state.setdefault("rt_frame_i", 0)
	st.session_state["rt_frame_i"] += 1

	# Every 5th frame: run detection; otherwise return last annotated (if any) or raw frame
	should_infer = (st.session_state["rt_frame_i"] % 5 == 0)

	if should_infer:
		try:
			image_resized = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
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
				detections = [
					Detection(
						class_id=int(_box.cls),
						label=CLASSES[int(_box.cls)],
						score=float(_box.conf),
						box=_box.xyxy[0].astype(int),
					)
					for _box in boxes
				]
				result_queue.put(detections)
				for d in detections:
					append_detection_log(
						source="realtime",
						class_label=d.label,
						confidence=d.score,
						x1=int(d.box[0]),
						y1=int(d.box[1]),
						x2=int(d.box[2]),
						y2=int(d.box[3]),
						latitude=latitude,
						longitude=longitude,
						media_name=None,
					)
			annotated_frame = results[0].plot()
			_image = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation=cv2.INTER_AREA)
			st.session_state["rt_last_ann"] = _image
			return av.VideoFrame.from_ndarray(_image, format="bgr24")
		except Exception:
			pass

	# Fallback: return last annotated frame if available, else raw camera frame
	last = st.session_state.get("rt_last_ann")
	if isinstance(last, np.ndarray) and last.shape[:2] == (h_ori, w_ori):
		return av.VideoFrame.from_ndarray(last, format="bgr24")
	return av.VideoFrame.from_ndarray(image, format="bgr24")


st.markdown("#### Camera")
preview_only = st.toggle("Preview camera (no detection)", value=False, key="preview_toggle")

if preview_only:
	def preview_callback(frame: av.VideoFrame) -> av.VideoFrame:
		return frame

	webrtc_ctx = webrtc_streamer(
		key="road-damage-detection-preview",
		mode=WebRtcMode.SENDRECV,
		video_frame_callback=preview_callback,
		# Host-only ICE to avoid STUN issues on restricted networks
		rtc_configuration={"iceServers": []},
		media_stream_constraints={
			"video": True,
			"audio": False,
		},
		# show device selector and manual start
		desired_playing_state=None,
		translations={
			"start": "Start",
			"stop": "Stop",
			"select_device": "Select device",
		},
		video_html_attrs={
			"playsinline": True,
			"muted": True,
		},
		video_receiver_size=1,
		async_processing=False,
	)
else:
	webrtc_ctx = webrtc_streamer(
		key="road-damage-detection",
		mode=WebRtcMode.SENDRECV,
		video_frame_callback=video_frame_callback,
		# Host-only ICE to avoid STUN issues on restricted networks
		rtc_configuration={"iceServers": []},
		# Keep constraints minimal; user will pick device from selector
		media_stream_constraints={
			"video": True,
			"audio": False,
		},
		# do not auto-start so the device selector is visible
		desired_playing_state=None,
		translations={
			"start": "Start",
			"stop": "Stop",
			"select_device": "Select device",
		},
		video_html_attrs={
			"playsinline": True,
			"muted": True,
		},
		video_receiver_size=1,
		async_processing=True,
	)

# Inference controls
st.session_state["rt_conf"] = st.slider("Confidence", min_value=0.0, max_value=1.0, value=0.02, step=0.05, key="rt_conf_slider")
st.session_state["rt_iou"] = st.slider("IOU (NMS)", min_value=0.1, max_value=0.95, value=float(st.session_state.get("rt_iou", 0.7)), step=0.05, key="rt_iou_slider")
st.session_state["rt_imgsz"] = st.select_slider("Image size", options=[640, 960, 1280], value=int(st.session_state.get("rt_imgsz", 640)), key="rt_imgsz_slider")
st.session_state["rt_tta"] = st.checkbox("Enable TTA (Test-Time Augmentation)", value=bool(st.session_state.get("rt_tta", False)))

st.caption("Tip: Lower confidence to increase recall; reduce IOU to keep more overlapping boxes; increase image size for small cracks.")

st.divider()

if st.checkbox("Show Predictions Table", value=False):
	if webrtc_ctx.state.playing:
		labels_placeholder = st.empty()
		while True:
			result = result_queue.get()
			labels_placeholder.table(result)


 



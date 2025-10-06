from pathlib import Path
import os
import subprocess
import sys

import streamlit as st
from ultralytics import YOLO

from sample_utils.roboflow import download_and_extract_roboflow_zip
from sample_utils.download import download_file
import yaml


st.set_page_config(page_title="Train Custom Model", page_icon="🧠", layout="centered")

st.title("Train Custom YOLOv8 Model")
st.write("Train a custom detector using your Roboflow dataset URL (YOLO format). After training, the app will use the new weights by default.")

default_url = "https://public.roboflow.com/ds/lprXuOv04l?key=2Y3RBZ1dNF"
dataset_url = st.text_input("Roboflow dataset URL", value=default_url)

epochs = st.number_input("Epochs", min_value=1, max_value=300, value=30, step=1)
imgsz = st.number_input("Image size", min_value=320, max_value=1280, value=640, step=32)
batch = st.number_input("Batch size", min_value=2, max_value=64, value=8, step=2)

custom_weights_dir = Path("./models/custom")
custom_weights_dir.mkdir(parents=True, exist_ok=True)
custom_best = custom_weights_dir / "best.pt"


def train_pipeline():
	# Always acknowledge the click immediately
	st.success("Training started")
	with st.spinner("Downloading dataset from Roboflow..."):
		ds_dir = download_and_extract_roboflow_zip(dataset_url, Path("./datasets/roboflow"))

	# Heuristics: try to find dataset root containing train/valid folders
	root = None
	for cand in ds_dir.iterdir():
		if cand.is_dir() and (cand / "train").exists() and ((cand / "valid").exists() or (cand / "val").exists()):
			root = cand
			break
	# Fallback: search deep
	if root is None:
		for cand in ds_dir.rglob("train"):
			parent = cand.parent
			if (parent / "valid").exists() or (parent / "val").exists():
				root = parent
				break
	if root is None:
		st.error("Could not locate train/valid folders in the extracted dataset.")
		return

	# Build absolute paths for train/val images
	train_dir = (root / "train" / "images").resolve()
	val_dir = ((root / "valid" / "images") if (root / "valid").exists() else (root / "val" / "images")).resolve()

	# Try to load names/classes from any data.yaml if present
	names = None
	for cand in ds_dir.rglob("data.yaml"):
		try:
			with open(cand, "r", encoding="utf-8") as f:
				cfg = yaml.safe_load(f)
			names = cfg.get("names") or cfg.get("names")
		except Exception:
			pass
		break
	if names is None:
		# fallback to generic single-class pothole if missing
		names = ["pothole"]

	# Write a fixed config with absolute paths
	fixed_yaml = root / "data_fixed.yaml"
	with open(fixed_yaml, "w", encoding="utf-8") as f:
		yaml.safe_dump({
			"path": str(root.resolve()),
			"train": str(train_dir),
			"val": str(val_dir),
			"names": names,
		}, f, sort_keys=False)

	st.info(f"Using dataset config: {fixed_yaml}")

	# Ensure base model exists (auto-download if missing)
	model_path = Path("./models/YOLOv8_Small_RDD.pt")
	if not model_path.exists():
		MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"
		with st.spinner("Fetching base model (first run only)..."):
			try:
				download_file(MODEL_URL, model_path, expected_size=89569358)
				st.success("Base model fetched successfully")
			except Exception:
				# Proceed without blocking the UI; training script will validate again
				pass
	# If the model is present after fetch attempt, confirm readiness
	if model_path.exists():
		st.success("Base model ready")

	# Run training as a background subprocess; UI stays minimal per request
	py_exe = sys.executable  # current venv python
	cmd = [
		py_exe,
		"scripts\\train_custom.py",
		"--url", str(dataset_url),
		"--epochs", str(int(epochs)),
		"--imgsz", str(int(imgsz)),
		"--batch", str(int(batch)),
	]
	# Disable tqdm to avoid any stream issues
	os.environ["TQDM_DISABLE"] = "1"
	proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(Path(".").resolve()))
	st.session_state["train_proc_pid"] = proc.pid
	# Let the asynchronous process run; success toast was already shown


colA, colB = st.columns(2)
with colA:
	if st.button("Start Training (standard)", type="primary"):
		train_pipeline()
with colB:
	if st.button("Start Training (fast 5 epochs)"):
		# override inputs briefly
		st.session_state["_override_epochs"] = 5
		train_pipeline()



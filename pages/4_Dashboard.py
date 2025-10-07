from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
	page_title="Analytics Dashboard",
	page_icon="📊",
	layout="wide",
)

st.title("Road Damage Analytics Dashboard")
st.caption("Aggregates detections across realtime, image, and video pages with filters, charts, and exports.")

data_path = Path("./data/detections.csv")

if not data_path.exists():
	st.info("No detections logged yet. Run any detection page to generate data.")
else:
	# Load all data
	df = pd.read_csv(data_path)
	# Coerce dtypes and enrich time columns
	if "timestamp_iso" in df.columns:
		try:
			df["timestamp"] = pd.to_datetime(df["timestamp_iso"], errors="coerce")
		except Exception:
			df["timestamp"] = pd.NaT
		df["date"] = df["timestamp"].dt.date
		df["hour"] = df["timestamp"].dt.hour

	# Sidebar filters
	st.sidebar.header("Filters")
	# Source filter
	_sources = sorted([s for s in df["source"].dropna().unique().tolist()]) if "source" in df else []
	selected_sources = st.sidebar.multiselect("Sources", options=_sources, default=_sources)
	# Class filter
	_classes = sorted([c for c in df["class_label"].dropna().unique().tolist()]) if "class_label" in df else []
	selected_classes = st.sidebar.multiselect("Classes", options=_classes, default=_classes)
	# Date range filter
	if "timestamp" in df:
		min_date = pd.to_datetime(df["timestamp"].min()).date() if df["timestamp"].notna().any() else None
		max_date = pd.to_datetime(df["timestamp"].max()).date() if df["timestamp"].notna().any() else None
		if min_date and max_date:
			start_date, end_date = st.sidebar.date_input("Date range", value=(min_date, max_date))
		else:
			start_date, end_date = None, None
	else:
		start_date, end_date = None, None

	# Optional live auto-refresh
	st.sidebar.divider()
	live = st.sidebar.toggle("Live updates", value=True, help="Auto-refresh analytics periodically")
	interval = st.sidebar.slider("Refresh interval (sec)", 3, 60, 5) if live else None
	if live:
		# Lightweight client-side reload to re-read CSV without blocking
		components.html(
			f"""
			<script>
			setTimeout(function() {{ location.reload(); }}, {interval * 1000});
			</script>
			""",
			height=0,
		)

	filtered = df.copy()
	if selected_sources:
		filtered = filtered[filtered["source"].isin(selected_sources)]
	if selected_classes:
		filtered = filtered[filtered["class_label"].isin(selected_classes)]
	if start_date and end_date and "date" in filtered:
		# Compare date objects directly (both sides are datetime.date)
		filtered = filtered[(filtered["date"] >= start_date) & (filtered["date"] <= end_date)]

	# KPIs over filtered set
	st.subheader("Overview (filtered)")
	col1, col2, col3, col4 = st.columns(4)
	with col1:
		st.metric("Total Detections", int(len(filtered)))
	with col2:
		st.metric("Unique Classes", int(filtered["class_label"].nunique() if "class_label" in filtered else 0))
	with col3:
		st.metric("Avg Confidence", f"{filtered['confidence'].mean():.2f}" if "confidence" in filtered and len(filtered) else "-")
	with col4:
		st.metric("With GPS", int(filtered.dropna(subset=["latitude", "longitude"]).shape[0]) if {"latitude","longitude"}.issubset(filtered.columns) else 0)

	# Breakdown charts
	colA, colB = st.columns(2)
	with colA:
		st.subheader("Detections by Class")
		if "class_label" in filtered and not filtered.empty:
			st.bar_chart(filtered["class_label"].value_counts().sort_values(ascending=False))
	with colB:
		st.subheader("Detections by Source")
		if "source" in filtered and not filtered.empty:
			st.bar_chart(filtered["source"].value_counts())

	# Time series
	if "date" in filtered and not filtered.empty:
		st.subheader("Detections Over Time")
		_counts = filtered.groupby("date").size().rename("count").reset_index()
		_counts = _counts.sort_values("date")
		st.line_chart(_counts.set_index("date")["count"])

	# Confidence distribution
	if "confidence" in filtered and not filtered.empty:
		st.subheader("Confidence Distribution")
		st.area_chart(filtered["confidence"].clip(0, 1))

	# Map over filtered set
	if {"latitude", "longitude"}.issubset(filtered.columns):
		_geo = filtered.dropna(subset=["latitude", "longitude"])  # keep rows with coordinates
		if len(_geo) > 0:
			st.subheader("Map of Detections (filtered)")
			st.map(_geo.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]])

	# Recent records table and exports
	st.subheader("Records")
	st.caption("Showing the latest 200 for performance; download for full dataset.")
	preview = filtered.tail(200)
	st.dataframe(preview, use_container_width=True)

	colD, colE = st.columns(2)
	with colD:
		st.download_button("Download Filtered CSV", data=filtered.to_csv(index=False).encode("utf-8"), file_name="detections_filtered.csv")
	with colE:
		st.download_button("Download Full CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="detections_all.csv")



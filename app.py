
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Construction Object Detection",
    page_icon="🏗️",
    layout="wide"
)

# Application title
st.title("🏗️ Construction Object Detection")

st.write(
    "Upload construction site images or videos "
    "to detect objects using your trained YOLO11 model."
)

# Load trained YOLO model
@st.cache_resource
def load_model():
    return YOLO("naga.pt")

model = load_model()

# Confidence threshold
confidence = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05
)

# Input type selection
input_type = st.radio(
    "Select Input Type",
    ["Image", "Video"],
    horizontal=True
)


# =====================================
# IMAGE DETECTION
# =====================================

if input_type == "Image":

    uploaded_file = st.file_uploader(
        "Upload Construction Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        # Convert image to NumPy
        image_array = np.array(image)

        # YOLO prediction
        results = model.predict(
            source=image_array,
            conf=confidence,
            verbose=False
        )

        # Annotated image
        annotated_image = results[0].plot()
        annotated_image = cv2.cvtColor(
            annotated_image,
            cv2.COLOR_BGR2RGB
        )

        with col2:
            st.subheader("Detection Results")
            st.image(
                annotated_image,
                use_container_width=True
            )

        # Detection details
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:

            object_names = []
            scores = []

            for box in boxes:

                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                score = float(box.conf[0])

                object_names.append(class_name)
                scores.append(score)

            # Calculate metrics
            total_objects = len(object_names)
            avg_score = np.mean(scores) * 100
            max_score = max(scores) * 100

            st.subheader("📊 Detection Score Summary")

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Objects", total_objects)
            col2.metric("Average Confidence", f"{avg_score:.2f}%")
            col3.metric("Highest Confidence", f"{max_score:.2f}%")

            # Detection table
            st.subheader("Detected Object Details")

            df = pd.DataFrame({
                "Object Name": object_names,
                "Confidence Score (%)": [
                    round(s * 100, 2) for s in scores
                ]
            })

            st.dataframe(df, use_container_width=True)

            # Class count
            st.subheader("Class-wise Detection Count")

            class_counts = df["Object Name"].value_counts()

            st.bar_chart(class_counts)

        else:
            st.warning("No objects detected.")


# =====================================
# VIDEO DETECTION
# =====================================

elif input_type == "Video":

    uploaded_video = st.file_uploader(
        "Upload Construction Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        # Save uploaded video temporarily
        temp_input = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_input.write(uploaded_video.read())
        temp_input.close()

        input_path = temp_input.name

        cap = cv2.VideoCapture(input_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        # Output video
        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        output_path = output_file.name
        output_file.close()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps if fps > 0 else 25,
            (width, height)
        )

        st.subheader("Video Processing")

        progress_bar = st.progress(0)
        status_text = st.empty()

        frame_count = 0

        # Track overall detection information
        all_scores = []
        all_names = []

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            # YOLO detection
            results = model.predict(
                source=frame,
                conf=confidence,
                verbose=False
            )

            # Annotate frame
            annotated_frame = results[0].plot()

            out.write(annotated_frame)

            # Collect detection scores
            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:

                for box in boxes:

                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    score = float(box.conf[0])

                    all_names.append(class_name)
                    all_scores.append(score)

            frame_count += 1

            progress = min(
                frame_count / max(total_frames, 1),
                1.0
            )

            progress_bar.progress(progress)

            status_text.write(
                f"Processing frame {frame_count}/{total_frames}"
            )

        cap.release()
        out.release()

        status_text.success("Video processing completed!")

        # Display processed video
        st.subheader("🎬 Detection Output Video")

        st.video(output_path)

        # Video detection metrics
        st.subheader("📊 Video Detection Summary")

        if len(all_scores) > 0:

            total_detections = len(all_scores)
            avg_score = np.mean(all_scores) * 100
            max_score = max(all_scores) * 100

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Detections",
                total_detections
            )

            col2.metric(
                "Average Confidence",
                f"{avg_score:.2f}%"
            )

            col3.metric(
                "Highest Confidence",
                f"{max_score:.2f}%"
            )

            # Video detection table
            video_df = pd.DataFrame({
                "Object Name": all_names,
                "Confidence Score (%)": [
                    round(s * 100, 2)
                    for s in all_scores
                ]
            })

            st.subheader("Detected Object Details")

            st.dataframe(
                video_df,
                use_container_width=True
            )

            # Class-wise count
            st.subheader("Class-wise Detection Count")

            st.bar_chart(
                video_df["Object Name"].value_counts()
            )

        else:

            st.warning("No objects detected in this video.")

        # Download processed video
        with open(output_path, "rb") as video_file:

            st.download_button(
                label="⬇️ Download Detection Video",
                data=video_file,
                file_name="construction_detection.mp4",
                mime="video/mp4"
            )

        # Cleanup
        os.remove(input_path)
        os.remove(output_path)

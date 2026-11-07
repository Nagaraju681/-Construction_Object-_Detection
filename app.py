
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Construction Object Detection",
    page_icon="🏗️",
    layout="wide"
)

# Application title
st.title("🏗️ Construction Object Detection")

st.write(
    "Upload a construction site image to detect objects "
    "using your trained YOLO11 model."
)

# Load trained model
@st.cache_resource
def load_model():
    model = YOLO("naga.pt")
    return model

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# Confidence threshold
confidence = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05
)

if uploaded_file is not None:

    # Read uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    # Convert image to NumPy array
    image_array = np.array(image)

    # Perform object detection
    results = model.predict(
        source=image_array,
        conf=confidence
    )

    # Get annotated image
    annotated_image = results[0].plot()

    # Convert BGR to RGB
    annotated_image = annotated_image[:, :, ::-1]

    # Display detected image
    st.subheader("Detection Results")

    st.image(
        annotated_image,
        caption="Detected Objects",
        use_container_width=True
    )

    # Detection summary
    st.subheader("📊 Detection Score Summary")

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:

        object_names = []
        confidence_scores = []

        for box in boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence_score = float(box.conf[0])

            object_names.append(class_name)
            confidence_scores.append(confidence_score)

        # Calculate scores
        total_objects = len(object_names)

        average_confidence = (
            sum(confidence_scores) / total_objects
        )

        highest_confidence = max(confidence_scores)

        # Display metrics
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Objects Detected",
            total_objects
        )

        col2.metric(
            "Average Confidence",
            f"{average_confidence * 100:.2f}%"
        )

        col3.metric(
            "Highest Confidence",
            f"{highest_confidence * 100:.2f}%"
        )

        # Individual object details
        st.subheader("🔍 Detected Object Details")

        detection_data = pd.DataFrame({
            "Object Name": object_names,
            "Confidence Score (%)": [
                round(score * 100, 2)
                for score in confidence_scores
            ]
        })

        st.dataframe(
            detection_data,
            use_container_width=True,
            hide_index=True
        )

        # Class-wise object count
        st.subheader("📦 Class-wise Detection Count")

        class_counts = detection_data["Object Name"].value_counts()

        st.bar_chart(class_counts)

    else:

        st.warning("No objects detected in this image.")

else:

    st.info("Please upload an image to start detection.")

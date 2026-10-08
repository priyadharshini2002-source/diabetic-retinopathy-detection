import streamlit as st
import numpy as np
from PIL import Image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-top: 15px;
    }

    .disclaimer {
        font-size: 13px;
        color: #666;
        padding: 15px;
        border-radius: 8px;
        background-color: #f5f5f5;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_model():

    model = Sequential()

    # ----- Convolution Block 1 -----
    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(128, 128, 3)
        )
    )

    model.add(MaxPooling2D())
    model.add(BatchNormalization())

    # ----- Convolution Block 2 -----
    model.add(
        Conv2D(
            64,
            (3, 3),
            activation="relu"
        )
    )

    model.add(MaxPooling2D())
    model.add(BatchNormalization())

    # ----- Convolution Block 3 -----
    model.add(
        Conv2D(
            64,
            (3, 3),
            activation="relu"
        )
    )

    model.add(MaxPooling2D())
    model.add(BatchNormalization())

    # ----- Convolution Block 4 -----
    model.add(
        Conv2D(
            96,
            (3, 3),
            activation="relu"
        )
    )

    model.add(MaxPooling2D())
    model.add(BatchNormalization())

    # ----- Convolution Block 5 -----
    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu"
        )
    )

    model.add(MaxPooling2D())
    model.add(BatchNormalization())

    # ----- Fully Connected Layers -----
    model.add(Dropout(0.2))
    model.add(Flatten())

    model.add(
        Dense(
            128,
            activation="relu"
        )
    )

    model.add(Dropout(0.3))

    # ----- Output Layer -----
    model.add(
        Dense(
            4,
            activation="softmax"
        )
    )

    # Load trained weights
    model.load_weights("model1.h5")

    return model


# Load model
model = load_model()


# =========================================================
# CLASS LABELS
# =========================================================

LABELS = [
    "Mild",
    "Moderate",
    "No_DR",
    "Severe"
]

IMAGE_SIZE = (128, 128)


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_retinopathy(image):

    # Resize image
    image = image.resize(IMAGE_SIZE)

    # Convert image to NumPy array
    image_array = np.array(image)

    # Convert grayscale image to RGB
    if len(image_array.shape) == 2:
        image_array = np.stack(
            (image_array,) * 3,
            axis=-1
        )

    # Remove alpha channel if present
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Generate prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )[0]

    # Get predicted class
    predicted_index = np.argmax(prediction)

    predicted_class = LABELS[predicted_index]

    # Get confidence
    confidence = prediction[predicted_index] * 100

    # Class-wise probabilities
    probabilities = {
        LABELS[i]: float(prediction[i] * 100)
        for i in range(len(LABELS))
    }

    return (
        predicted_class,
        confidence,
        probabilities
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🩺 Diabetic Retinopathy Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'CNN-Based Retinal Image Classification'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a retinal fundus image and let the trained "
    "CNN model classify it into one of four diabetic "
    "retinopathy categories."
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📌 Model Information")

    st.write("**Model:** Convolutional Neural Network")
    st.write("**Input Size:** 128 × 128 × 3")
    st.write("**Output Classes:** 4")

    st.subheader("🏷️ Classes")

    st.write("• Mild")
    st.write("• Moderate")
    st.write("• No_DR")
    st.write("• Severe")

    st.divider()

    st.subheader("⚙️ Processing")

    st.write("1. Image Upload")
    st.write("2. Resize to 128 × 128")
    st.write("3. CNN Prediction")
    st.write("4. Class Probability")
    st.write("5. Predicted Category")

    st.divider()

    st.warning(
        "⚠️ This application is intended for "
        "educational and research purposes only. "
        "It is not a medical diagnosis or a substitute "
        "for professional medical advice."
    )


# =========================================================
# IMAGE UPLOAD
# =========================================================

st.subheader("📤 Upload Retinal Image")

uploaded_file = st.file_uploader(
    "Choose a retinal fundus image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# MAIN APPLICATION
# =========================================================

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Convert for display
    display_image = image.convert("RGB")

    col1, col2 = st.columns(
        [1, 1],
        gap="large"
    )

    # -----------------------------------------------------
    # LEFT COLUMN
    # -----------------------------------------------------

    with col1:

        st.subheader("🖼️ Uploaded Image")

        st.image(
            display_image,
            caption="Retinal Fundus Image",
            use_container_width=True
        )

        st.caption(
            f"Image size: {image.size[0]} × {image.size[1]} pixels"
        )


    # -----------------------------------------------------
    # RIGHT COLUMN
    # -----------------------------------------------------

    with col2:

        st.subheader("🔍 Analysis")

        analyze_button = st.button(
            "🚀 Analyze Image",
            use_container_width=True,
            type="primary"
        )

        if analyze_button:

            with st.spinner(
                "Analyzing retinal image..."
            ):

                (
                    predicted_class,
                    confidence,
                    probabilities
                ) = predict_retinopathy(image)

            st.success(
                "Analysis completed successfully!"
            )

            st.markdown(
                '<div class="prediction-box">',
                unsafe_allow_html=True
            )

            st.metric(
                label="Predicted Class",
                value=predicted_class
            )

            st.metric(
                label="Model Confidence",
                value=f"{confidence:.2f}%"
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

            st.divider()

            # ------------------------------------------------
            # PROBABILITY RESULTS
            # ------------------------------------------------

            st.subheader(
                "📊 Class Probabilities"
            )

            for label, probability in probabilities.items():

                st.write(
                    f"**{label}** — "
                    f"{probability:.2f}%"
                )

                st.progress(
                    int(
                        min(
                            max(probability, 0),
                            100
                        )
                    )
                )

else:

    st.info(
        "👆 Upload a retinal fundus image above "
        "to start the analysis."
    )


# =========================================================
# ABOUT SECTION
# =========================================================

st.divider()

with st.expander("ℹ️ About This Project"):

    st.write(
        """
        This project uses a Convolutional Neural Network (CNN)
        to classify retinal fundus images into four categories:

        • Mild diabetic retinopathy

        • Moderate diabetic retinopathy

        • No diabetic retinopathy

        • Severe diabetic retinopathy

        The trained model receives a retinal image, resizes it
        to 128 × 128 pixels, and produces probability scores
        for the four classes.
        """
    )


# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="disclaimer">

    <b>Medical Disclaimer:</b><br>
    This application is developed for educational and research
    purposes. The predictions generated by this model should not
    be considered a medical diagnosis. Always consult a qualified
    healthcare professional for clinical evaluation and treatment.

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FOOTER
# =========================================================

st.caption(
    "Diabetic Retinopathy Detection • "
    "CNN-Based Machine Learning Project"
)
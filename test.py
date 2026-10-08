import os
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json

# ============================================================
# 1. MODEL FILE PATHS
# ============================================================

MODEL_JSON = "model1.json"
MODEL_WEIGHTS = "model1.h5"

# Test image directory
TEST_PATH = "data/test"

# Image size expected by the model
IMAGE_SIZE = (128, 128)

# Class labels
LABELS = ["Mild", "Moderate", "No_DR", "Severe"]


# ============================================================
# 2. LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_JSON):
    raise FileNotFoundError(
        f"Model architecture file not found: {MODEL_JSON}"
    )

if not os.path.exists(MODEL_WEIGHTS):
    raise FileNotFoundError(
        f"Model weights file not found: {MODEL_WEIGHTS}"
    )

print("Loading model...")

with open(MODEL_JSON, "r") as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)

# Load trained weights
loaded_model.load_weights(MODEL_WEIGHTS)

print("Loaded model from disk successfully.")
print()


# ============================================================
# 3. CLASSIFY ONE IMAGE
# ============================================================

def classify(img_file):
    """
    Load one image and predict its diabetic retinopathy class.
    """

    try:
        print("Image:", img_file)

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------
        test_image = image.load_img(
            img_file,
            target_size=IMAGE_SIZE
        )

        # Convert image to NumPy array
        test_image = image.img_to_array(test_image)

        # Add batch dimension
        # Shape becomes: (1, 128, 128, 3)
        test_image = np.expand_dims(test_image, axis=0)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------
        result = loaded_model.predict(test_image, verbose=0)

        # Get the prediction array
        probabilities = result[0]

        print("Raw model output:", probabilities)

        # ----------------------------------------------------
        # Check output size
        # ----------------------------------------------------
        if len(probabilities) != len(LABELS):
            print(
                "ERROR: Model output has",
                len(probabilities),
                "classes, but",
                len(LABELS),
                "labels were provided."
            )
            return None

        # ----------------------------------------------------
        # Find class with highest probability
        # ----------------------------------------------------
        predicted_index = np.argmax(probabilities)

        prediction = LABELS[predicted_index]

        confidence = probabilities[predicted_index] * 100

        # ----------------------------------------------------
        # Display individual probabilities
        # ----------------------------------------------------
        print("\nClass probabilities:")

        for label, probability in zip(LABELS, probabilities):
            print(
                f"  {label:10s}: {probability * 100:.2f}%"
            )

        print("\nPrediction:", prediction)
        print(f"Confidence: {confidence:.2f}%")

        return prediction, confidence

    except Exception as e:
        print("Error processing image:", img_file)
        print("Error:", e)
        return None


# ============================================================
# 4. FIND ALL TEST IMAGES
# ============================================================

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(
        f"Test directory not found: {TEST_PATH}"
    )

files = []

# Supported image extensions
valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)

for root, directories, filenames in os.walk(TEST_PATH):

    for filename in filenames:

        if filename.lower().endswith(valid_extensions):

            full_path = os.path.join(root, filename)

            files.append(full_path)


# ============================================================
# 5. CHECK IF IMAGES WERE FOUND
# ============================================================

print("Test directory:", TEST_PATH)
print("Number of images found:", len(files))
print()

if len(files) == 0:
    print("No image files were found.")
    print("Make sure your images are inside:")
    print(os.path.abspath(TEST_PATH))

else:

    print("=" * 60)
    print("STARTING PREDICTION")
    print("=" * 60)
    print()

    # Store results
    predictions = []

    # ========================================================
    # 6. CLASSIFY EACH IMAGE
    # ========================================================

    for file_path in files:

        result = classify(file_path)

        if result is not None:

            prediction, confidence = result

            predictions.append({
                "image": file_path,
                "prediction": prediction,
                "confidence": confidence
            })

        print()
        print("-" * 60)
        print()


# ============================================================
# 7. DISPLAY FINAL RESULTS
# ============================================================

if len(predictions) > 0:

    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    for result in predictions:

        print(
            f"{result['image']}  -->  "
            f"{result['prediction']}  "
            f"({result['confidence']:.2f}%)"
        )


# ============================================================
# 8. DISPLAY SUMMARY
# ============================================================

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

if len(predictions) > 0:

    # Count predictions for each class
    class_counts = {
        label: 0 for label in LABELS
    }

    for result in predictions:

        prediction = result["prediction"]

        if prediction in class_counts:
            class_counts[prediction] += 1

    print("Total images processed:", len(predictions))
    print()

    for label in LABELS:

        print(
            f"{label:10s}: "
            f"{class_counts[label]}"
        )

else:

    print("No images were successfully classified.")

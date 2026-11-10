
# Import FastAPI to create the API application.
# UploadFile and File allow the API to receive uploaded image files.
from fastapi import FastAPI, UploadFile, File

# Import CORS middleware to allow requests from the frontend.
from fastapi.middleware.cors import CORSMiddleware

# Import OpenCV for image processing and face detection.
import cv2

# Import NumPy for converting image bytes into arrays and handling model input.
import numpy as np

# Import load_model to load the trained TensorFlow/Keras emotion model.
from tensorflow.keras.models import load_model


# Create the FastAPI application and set its title.
app = FastAPI(title="PlaceReady Emotion (FER-2013)")


# Configure CORS so frontend applications can send requests to this API.
app.add_middleware(
    CORSMiddleware,

    # Allow requests from all origins.
    # For production, replace "*" with your trusted frontend URL.
    allow_origins=["*"],

    # Allow requests that include credentials.
    allow_credentials=True,

    # Allow all HTTP methods, such as GET and POST.
    allow_methods=["*"],

    # Allow all request headers.
    allow_headers=["*"],
)


# Define the path of the trained facial emotion recognition model.
MODEL_PATH = "face_model.h5"

# Load the saved model without compiling it because training is not performed here.
model = load_model(MODEL_PATH, compile=False)


# Print a separator to make the startup information easy to read.
print("========================================")

# Display a message confirming that the model has loaded.
print("Emotion Model Loaded")

# Display the model's expected input dimensions.
# These dimensions determine the size and format of images sent to the model.
print("Model input shape :", model.input_shape)

# Display the model's output dimensions.
# The output represents the scores predicted for the emotion categories.
print("Model output shape:", model.output_shape)

# Print another separator.
print("========================================")


# Define the seven emotion labels used by the trained model.
# The order must match the class order used when training the model.
EMOTIONS = [
    "angry",       # Anger
    "disgust",     # Disgust
    "fearful",     # Fear
    "happy",       # Happiness
    "sad",         # Sadness
    "surprised",   # Surprise
    "neutral",     # Neutral expression
]


# Load OpenCV's pre-trained Haar Cascade face detector.
# cv2.data.haarcascades provides the directory containing OpenCV's XML files.
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# Create a GET endpoint to check whether the API is responding.
@app.get("/health")
def health():

    # Return the API status and the model's input/output dimensions as JSON.
    return {
        # Indicates that the health endpoint responded successfully.
        "ok": True,

        # Indicates that model loading is expected to have succeeded.
        # This is hardcoded; it does not independently verify model availability.
        "model_loaded": True,

        # Return the input dimensions as integers.
        # Preserve None for any dimension that is unspecified.
        "model_input": [
            int(x) if x is not None else None
            for x in model.input_shape
        ],

        # Return the output dimensions using the same conversion.
        "model_output": [
            int(x) if x is not None else None
            for x in model.output_shape
        ],
    }


# Create a POST endpoint at /analyze to process an uploaded image.
# The frontend sends the image using the form field named "file".
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    # Read the uploaded image file into bytes.
    contents = await file.read()

    # Convert the image bytes into a NumPy array that OpenCV can process.
    nparr = np.frombuffer(contents, np.uint8)

    # Decode the array into an image using OpenCV's color image format.
    # The resulting image uses BGR color-channel order.
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Check whether OpenCV successfully decoded the uploaded file.
    if img is None:

        # Return an empty result and an error message for an invalid image.
        return {
            "faces": [],
            "dominant": None,
            "face_count": 0,
            "error": "Invalid image file",
        }

    # Convert the color image into grayscale.
    # Face detection can work with grayscale images.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image using the Haar Cascade detector.
    faces = face_cascade.detectMultiScale(
        gray,

        # Reduce the image size by this factor at each detection scale.
        scaleFactor=1.2,

        # Require multiple neighboring detections to help filter false positives.
        minNeighbors=4,

        # Ignore detected regions smaller than 40 by 40 pixels.
        minSize=(40, 40)
    )

    # Create an empty list to store the results for each detected face.
    results = []

    # Get the input shape expected by the trained emotion model.
    input_shape = model.input_shape

    # Extract the required image height from the model's input shape.
    target_height = int(input_shape[1])

    # Extract the required image width from the model's input shape.
    target_width = int(input_shape[2])

    # Extract the number of input channels.
    # Use 1 if the model's channel dimension is unspecified.
    # A value of 1 means grayscale; 3 means RGB.
    channels = int(input_shape[3]) if input_shape[3] is not None else 1

    # Process each detected face.
    # x and y represent the top-left corner.
    # w and h represent the width and height of the face.
    for (x, y, w, h) in faces:

        # Crop the face region from the grayscale image.
        # ROI stands for Region of Interest.
        roi = gray[y : y + h, x : x + w]

        # Resize the cropped face to the dimensions expected by the model.
        roi = cv2.resize(roi, (target_width, target_height))

        # Convert pixel values from the range 0–255 to the range 0–1.
        # This matches a common preprocessing approach for image models.
        roi = roi.astype("float32") / 255.0

        # Prepare the image according to the model's expected channel count.
        if channels == 1:

            # Add a channel dimension to the grayscale image.
            # Shape changes from (height, width) to (height, width, 1).
            roi = np.expand_dims(roi, axis=-1)

        elif channels == 3:

            # Convert the normalized grayscale image back to 8-bit values.
            # Convert it into a three-channel RGB image.
            # Convert the result to float32 and normalize it to 0–1 again.
            roi = (
                cv2.cvtColor(
                    (roi * 255).astype("uint8"),
                    cv2.COLOR_GRAY2RGB
                ).astype("float32")
                / 255.0
            )

        else:

            # Return an error if the model expects an unsupported
            # number of image channels.
            return {
                "faces": [],
                "dominant": None,
                "face_count": 0,
                "error": f"Unsupported model channels: {channels}",
            }

        # Add a batch dimension because Keras expects batches of images.
        # Shape becomes (1, height, width, channels).
        roi = np.expand_dims(roi, axis=0)

        # Run the model on the prepared face image.
        # verbose=0 prevents prediction progress output.
        # [0] selects the prediction for the first (and only) image in the batch.
        preds = model.predict(roi, verbose=0)[0]

        # Find the index of the emotion with the highest predicted score.
        emotion_index = int(np.argmax(preds))

        # Convert the predicted index into its corresponding emotion label.
        # Use "neutral" as a fallback if the index exceeds the label list.
        label = (
            EMOTIONS[emotion_index]
            if emotion_index < len(EMOTIONS)
            else "neutral"
        )

        # Get the highest score returned by the model.
        # This is a model confidence score and may not be a calibrated probability.
        confidence = float(np.max(preds))

        # Apply a simple rule to reduce low-confidence "sad" predictions.
        # This is a heuristic, not a guarantee that the face is neutral.
        if label == "sad" and confidence < 0.55:
            label = "neutral"

        # If the highest prediction score is below 0.40,
        # classify the expression as neutral.
        if confidence < 0.40:
            label = "neutral"

        # Store the result for this face in the results list.
        results.append(
            {
                # Predicted facial-expression category.
                "emotion": label,

                # Highest score returned by the model.
                "confidence": confidence,

                # Bounding box of the detected face:
                # [x-coordinate, y-coordinate, width, height].
                "bbox": [int(x), int(y), int(w), int(h)],
            }
        )

    # Select the first detected face's expression as the dominant result.
    # If no faces were detected, return None.
    # Note: this does not compare all faces to find the most common expression.
    dominant = results[0]["emotion"] if results else None

    # Return all face results, the selected dominant expression,
    # and the total number of detected faces as JSON.
    return {
        "faces": results,
        "dominant": dominant,
        "face_count": len(results),
    }

import cv2
import numpy as np
import tensorflow as tf

# Load YOLOv3 model
model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
model.load_weights("yolov3_weights.h5")  # You should have YOLOv3 weights in this file

# Load YOLOv3 class names
with open("yolov3_classes.txt", "r") as f:
    classes = f.read().strip().split("\n")

# Function to preprocess image for YOLOv3
def preprocess_image(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

# Open a connection to the webcam
cap = cv2.VideoCapture(0)  # 0 represents the default camera (you can change this if you have multiple cameras)

while True:
    # Read frame from the webcam
    ret, frame = cap.read()

    # Preprocess the frame for YOLOv3
    preprocessed_frame = preprocess_image(frame)

    # Perform object detection with YOLOv3
    predictions = model.predict(preprocessed_frame)

    # Process YOLOv3 output
    boxes, scores, classes = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(predictions[..., :4], (tf.shape(predictions)[0], -1, 1, 4)),
        scores=tf.reshape(predictions[..., 4], (tf.shape(predictions)[0], -1, 1)),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.5,
        score_threshold=0.5
    )

    for box, score, class_id in zip(boxes[0], scores[0], classes[0]):
        if score > 0.5:
            # Object detected
            x, y, width, height = box.numpy()
            class_name = classes[class_id.numpy().astype(int)]
            confidence = score.numpy()

            # Draw a rectangle around the detected object
            cv2.rectangle(frame, (int(x), int(y)), (int(x + width), int(y + height)), (0, 255, 0), 2)

            # Add class label and confidence to the object
            label = f"{classes[class_id.numpy().astype(int)]}: {confidence:.2f}"
            cv2.putText(frame, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the image
    cv2.imshow("Object Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

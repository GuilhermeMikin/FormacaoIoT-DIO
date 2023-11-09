from ultralytics import YOLO
import cv2
import math 
from time import sleep
import sys

# Load the trained face recognition model
recognizer = cv2.face.EigenFaceRecognizer.create()
recognizer.read("my_Eigen_classifier.yml")  # Load your trained model

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("Frameworks_ML_IoT\Face_Recognition\classificadores\haarcascade-frontalface-default.xml")

label_to_name = {0: "she", 1: "me"}

object_list = list() # List of objects detected in the webcam

def recognize_person(face_roi):
    # Preprocess the face region (resize to match EigenFaceRecognizer's input size)
    face_roi = cv2.resize(face_roi, (100, 100))  # Adjust to the size used during training

    # Convert the face region to grayscale (if it's not already)
    face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
    # Use the face detection classifier to find faces
    faces = face_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Use EigenFaceRecognizer to predict the label for the face
    label, confidence = recognizer.predict(face_roi)
    print(f"cam label: {label}")

    # Determine if it's you or your girlfriend based on the label
    if label == 1:  # Replace with the label assigned to you
        return f"Gui: {confidence/100:.2f}"
    elif label == 0:  # Replace with the label assigned to your girlfriend
        return f"Amanda: {confidence/100:.2f}"
    else:
        return f"Unknown: {confidence/100:.2f}"

def process_video_frames():
    count = 0
    global object_list
    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values
                
                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100

                # class name
                cls = int(box.cls[0])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                label = f"{classNames[cls]}: {confidence:.2f}"

                # Check if it's a person
                if classNames[cls] == "person":
                    face_roi = img[y1:y2, x1:x2]
                    label = recognize_person(face_roi)
                    print(f"new label: {label}")
                
                cv2.putText(img, label, org, font, fontScale, color, thickness)

        # try:
        #     print(f"classname: {label} ")
        #     if count > 15 and float(confidence) >= 0.7 and object_list[-1] != label:
        #         object_list.append(label)
        #         count = 0
        #     elif count > 15 and float(confidence) >= 0.4 and object_list[-1] != label and object_list[-2] != label:
        #         object_list.append(label)
        #         count = 0
        # except Exception:
        #     print("Exception")
        #     object_list.append(label)
        # count += 1
        # print(f"count: {count}")
                
        sleep(.1)
        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    sys.exit()

if __name__ == "__main__":
    # Start webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # Model
    model = YOLO("yolo-Weights\yolov8n.pt")

    # Object classes
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
    
    process_video_frames()

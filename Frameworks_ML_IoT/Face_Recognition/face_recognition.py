import cv2

# Load the trained face recognition model
recognizer = cv2.face.EigenFaceRecognizer.create()
recognizer.read("my_classificadorEigen.yml")  # Load your trained model

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 for the default camera, adjust if needed

# Create a dictionary to map label indices to person names
label_to_name = {0: "she", 1: "me"}  # Update with your names and label indices

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier("Frameworks_ML_IoT\Face_Recognition\classificadores\haarcascade-frontalface-default.xml")

while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    # Convert the frame to grayscale for face recognition
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use the face detection classifier to find faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Resize the test image to match the size used during training
        gray = cv2.resize(gray, (100, 100)) 

        # Use the recognizer to predict the label (person) of the detected face
        label, confidence = recognizer.predict(gray)

        # Get the name associated with the predicted label
        predicted_name = label_to_name.get(label, "Unknown")

        # Draw a rectangle around the detected face and display the predicted name
        cv2.rectangle(frame, (x, y, w, h), (0, 255, 0), 2)
        
        cv2.putText(frame, predicted_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with the recognized face
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

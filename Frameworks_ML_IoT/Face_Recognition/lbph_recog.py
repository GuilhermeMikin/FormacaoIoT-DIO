import cv2

# LBPH - Local Binary Patterns Histograms

classifier = cv2.CascadeClassifier("classificadores\\haarcascade-frontalface-default.xml")
# LBPHFaceRecognizer::create(int radius = 1,
#                            int neighbors = 8,
#                            int grid_x = 8,
#                            int grid_y = 8,
#                            double	threshold = DBL_MAX)
recognizer = cv2.face.LBPHFaceRecognizer_create(threshold=2)
recognizer.read("classificadorLBPH.yml")

face_width = 100
face_height = 100

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

cap = cv2.VideoCapture(0) 

while (True):
    success, img = cap.read() 
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_faces = classifier.detectMultiScale(gray_img, scaleFactor=1.5, minSize=(30,30))

    for (x, y, l, a) in detected_faces:
        face_img = cv2.resize(gray_img[y:y + a, x:x + l], (face_width, face_height))
        cv2.rectangle(img, (x, y), (x + l, y + a), (0,0,255), 2)
        id, confidence = recognizer.predict(face_img)

        name = ""

        if id == 1:
            name = 'Carlos'
        elif id == 2:
            name = 'Amanda'
        elif id == 3:
            name = 'Luana'

        # cv2.putText(imagem, nome, (x0, y0 + altura do retângulo + tamanho da fonte mais espaço), fonte, tamanho da fonte, (R,G,B))
        cv2.putText(img, name, (x, y + a + 35), font, 2, (0, 0, 255))
        cv2.putText(img, str(confidence), (x, y + a + 55), font, 1, (0, 0, 255))


    cv2.imshow("Face", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
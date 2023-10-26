"""
Example of face, eye and mouth detection with the webcam
"""
import cv2

classifier = cv2.CascadeClassifier('Frameworks_ML_IoT//Face_Detection//classificadores//haarcascade-frontalface-default.xml')
# classifier = cv2.CascadeClassifier('Frameworks_ML_IoT//Face_Detection//classificadores//haarcascade-eye.xml')
# classifier = cv2.CascadeClassifier('Frameworks_ML_IoT//Face_Detection//classificadores//Mouth.xml')

# start webcam
cap = cv2.VideoCapture(0)

print("Pressione < q > para SAIR...")

while (True):
    conectado, face = cap.read()
    gray_imgs = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Imagem Cinza:", gray_imgs)

    found_faces = classifier.detectMultiScale(gray_imgs, scaleFactor=1.2, minSize=(100,100))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    for (x, y, l, a) in found_faces:
        cv2.rectangle(face, (x, y), (x + l, y + a), (0, 255, 0), 2)

    cv2.imshow("Webcam", face)

cap.release()
cv2.destroyAllWindows()
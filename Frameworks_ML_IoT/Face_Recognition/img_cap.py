import cv2
import numpy as np


classifier = cv2.CascadeClassifier("Face_Detection\\classificadores\\haarcascade-frontalface-default.xml")
eye_classifier = cv2.CascadeClassifier("Face_Detection\\classificadores\\haarcascade-eye.xml")

cap = cv2.VideoCapture(0)

light = 110

# Checks if the ambient lighting is adequate
while (True):
    success, img = cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converts color image to grayscale
    if np.average(gray_img) > light:
        print("\nA iluminação ambiente está adequada!")
        print("O valor deve ser maior do que %d." % light + " O valor ambiente é: %.2f" % np.average(gray_img))
        break
    else:
        print("\nO ambiente está muito escuro!")
        print("O valor deve ser maior do que %d." % light + " O valor ambiente é: %.2f" % np.average(gray_img))
        cv2.waitKey(2000) # Wait for 2 seconds


sample = 1 # Control number of samples
max_samples = 25 # Total number of samples
name = input('\nDigite um identificador (nome): ')
id = 1

face_width = 100 # img width
face_height = 100 # img heigth

eye_width = 30 # img width
eye_height = 30 # img heigth
print("Pressione a tecla < c > para capturar a foto:")


while (True):
    success, img = cap.read() 
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detected_faces = classifier.detectMultiScale(gray_img, scaleFactor=1.5, minSize=(face_width, face_height))

    for(x, y, l, a) in detected_faces:
        cv2.rectangle(img, (x, y), (x+l, y+a), (0, 0, 255), 2)
        region = img[y:y + a, x:x +l]

        gray_eye_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        detected_eyes = eye_classifier.detectMultiScale(gray_eye_region, scaleFactor=1.01, minNeighbors=20, minSize=(eye_width, eye_height))

        for (ox, oy, ol, oa) in detected_eyes:
            cv2.rectangle(region, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)


        if(cv2.waitKey(1) & 0xFF == ord('c')): # 'c' saves the img
            face_img = cv2.resize(gray_img[y:y + a, x:x +l], (face_width, face_height)) 
            cv2.imwrite("fotos/" + str(name) + "." + str(id) + "." + str(sample) + ".jpg", face_img)
            print("[foto " + str(name) + "." + str(id) + "." + str(sample) + " capturada com sucesso!")
            sample += 1
           

    cv2.imshow("Face", img)
    #cv2.waitKey(1) # Wait 1 second to process the img
    if (sample >= max_samples + 1):
           break 

print("Faces capturadas com sucesso!")
cap.release()
cv2.destroyAllWindows ()

import cv2
import os
import numpy as np

people = ['me', 'she']
data_folder = "Frameworks_ML_IoT//Face_Recognition//fotos"

classifier = cv2.CascadeClassifier("classificadores\\haarcascade-frontalface-default.xml")

features = []
labels = []

for person in people:
    path = data_folder + person
    #path = os.path.join(data_folder, person)
    print(data_folder, person, path)
    label = people.index(person)

    for img in os.listdir(person):
    #for img in os.listdir(path):
        img_path = os.path.join(path, img)

        img_array = cv2.imread(img_path)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        faces_rect = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        for (x,y,w,h) in faces_rect:
            faces_roi = gray[y:y+h, x:x+w]
            features.append(faces_roi)
            labels.append(label)


features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv2.face
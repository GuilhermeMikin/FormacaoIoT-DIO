import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create(40, 8000) # Creating eigenface classifier
fisherface = cv2.face.FisherFaceRecognizer_create(3, 2000) # Creating fisherface classifier
lbph = cv2.face.LBPHFaceRecognizer_create(2, 10) # Creating lbph classifier

def get_img_with_id():
    paths = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    faces = []
    ids = []
    for img_path in paths:
        face_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(img_path)[-1].split('.')[1]) # Get img id
        ids.append(id) # Create a img id vector
        faces.append(face_img) # Create a img vector matrix
        
       # cv.imshow("Face", imagemFace)
       # cv2.waitKey(10)
    return np.array(ids), faces

ids, faces = get_img_with_id()

print("Treinando...")
eigenface.train(faces, ids)
eigenface.write('classificadorEigen.yml')

fisherface.train(faces, ids)
fisherface.write('classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')

print("Treinamento realizado!")

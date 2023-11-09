"""
Authors: Fabio Vincenzi and Guilherme Balduino Lopes
This example uses the OpenCV (Open Source Computer Vision) library, created by Intel in 1999 and is written in C/C++import cv2
"""
import cv2

# Create a classifier using the information contained in the haarcascade-frontalface-default.xml file
classifier = cv2.CascadeClassifier('Frameworks_ML_IoT//Face_Detection//classificadores//haarcascade-frontalface-default.xml')

# faces = cv2.imread('Frameworks_ML_IoT//Face_Detection//fotos//eeu.jpeg') # It loads the model.jpg image and stores it in the faces variable
faces = cv2.imread('Frameworks_ML_IoT//Face_Detection//fotos//muitasFaces.jpg') # It loads the image many Faces.jpg and stores it in the faces variable
gray_img = cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY) # Converts color image to grayscale
                                                      # Before detecting a face, it is recommended to convert it to grayscale

cv2.imshow("Imagem Cinza:", gray_img) # Displays the image in grayscale

# Uses the detectMultiScale method to detect faces in the grayscale image
# The variable facesEncontradas will receive a matrix whose lines will be vectors containing the information [x,y,l,a]
# of faces found. Where: x and y are the initial positions of the face and l is the width and height of the faces found
# scaleFactor=1.1 changes the image scale, its value must always be greater than 1
# minNeighbors=20 lower values allow greater proximity between rectangles
# minSize=(width, height) minimum size of rectangles
found_faces = classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=20, minSize=(40,40))
print("Face(s) encontrada(s):",len(found_faces))


# Scan the vector matrix (matrix rows), vector by vector
# Example, considering that the image has 4 sides and that
#facesFound = [[ 83 65 110 110] [432 91 109 109] [242 151 99 99] [170 40 101 101]]
# in the first interaction x=83, y=65, l=110, a=110
# in the second interaction x=432, y=91, l=109, a=109
# in the third interaction x=242, y=151, l=99, a=99
# in the fourth interaction x=170, y=40, l=101, a=101
for (x, y, l, a) in found_faces:
    # cv2.rectangle(faces, (x0, y0), (x0 + largura, y0 + altura), (R, G, B), LarguraBorda)
    cv2.rectangle(faces, (x, y), (x + l, y + a), (0, 255, 0), 2)

cv2.imshow("Face(s) encontrada(s):", faces)
cv2.waitKey() # Wait until a key is pressed
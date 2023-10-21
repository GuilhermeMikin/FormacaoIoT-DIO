import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create(40, 8000) # Criando o classificador eigenface
fisherface = cv2.face.FisherFaceRecognizer_create(3, 2000) # Criando o classificador fisherface
lbph = cv2.face.LBPHFaceRecognizer_create(2, 10) # Criando o classificador lbph

def getImagemComId():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')] # Cria um vetor com o caminho de todas as imagens e armazena na variável caminhos
    faces = []
    ids = []
    for caminhoImagem in caminhos: # Varre o vetor de caminhos das imagens
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1]) # Pega o id da imagem
        ids.append(id) # Cria um vetor de ids das imagens
        faces.append(imagemFace) # Cria uma matriz de vetores de imagens
        
       # cv.imshow("Face", imagemFace)
       # cv2.waitKey(10)
    return np.array(ids), faces

ids, faces = getImagemComId()

print("Treinando...")
eigenface.train(faces, ids)
eigenface.write('classificadorEigen.yml')

fisherface.train(faces, ids)
fisherface.write('classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')

print("Treinamento realizado!")

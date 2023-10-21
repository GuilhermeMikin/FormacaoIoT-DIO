import cv2 # biblioteca para fazer o reconhecimento
import numpy as np # biblioteca para arrays e matrizes multidimensionais

# Cria um classificador utilizando as informações conditas no arquivo haarcascade-frontalface-default.xml
classificador = cv2.CascadeClassifier("Face_Detection\\classificadores\\haarcascade-frontalface-default.xml")
# Cria um classificador utilizando as informações conditas no arquivo haarcascade-eye.xml
classificadorOlho = cv2.CascadeClassifier("Face_Detection\\classificadores\\haarcascade-eye.xml")

camera = cv2.VideoCapture(0) # Captura o video da câmera

luz = 110

# Verifica se a iluminação ambiente está adequada
while (True):
    conectado, imagem = camera.read()  # Pega a leitura da camera
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)  # Transforma a imagem em cinza
    if np.average(imagemCinza) > luz:
        print("\nA iluminação ambiente está adequada!")
        print("O valor deve ser maior do que %d." % luz + " O valor ambiente é: %.2f" % np.average(imagemCinza))
        break
    else:
        print("\nO ambiente está muito escuro!")
        print("O valor deve ser maior do que %d." % luz + " O valor ambiente é: %.2f" % np.average(imagemCinza))
        cv2.waitKey(2000) # Espera 2s


amostra = 1 # Controla o numero de amostras
numeroAmostras = 25 # Numero total de amostra
nome = input('\nDigite um identificador (nome): ') # Nome da pessoa
id = 1

larguraFace = 100 # Largura da imagem
alturaFace = 100 # Altura da imagem

larguraOlho = 30 # Largura da imagem
alturaOlho = 30 # Altura da imagem
print("Pressione a tecla < c > para capturar a foto:")



while (True):
    conectado, imagem = camera.read() # Pega a leitura da camera
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # Transforma a imagem em cinza

    facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(larguraFace, alturaFace))

    for(x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x+l, y+a), (0, 0, 255), 2)
        regiao = imagem[y:y + a, x:x +l]

        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        olhosDetectados = classificadorOlho.detectMultiScale(regiaoCinzaOlho, scaleFactor=1.01, minNeighbors=20, minSize=(larguraOlho, alturaOlho))

        for (ox, oy, ol, oa) in olhosDetectados:
            cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)


        if(cv2.waitKey(1) & 0xFF == ord('c')): # Quando aperta a tecla 'c' salva a imagem que capturada pela webcam
            imagemFace = cv2.resize(imagemCinza[y:y + a, x:x +l], (larguraFace, alturaFace)) # Redimencionar a imagem
            cv2.imwrite("fotos/" + str(nome) + "." + str(id) + "." + str(amostra) + ".jpg", imagemFace) # Salva a foto
            print("[foto " + str(nome) + "." + str(id) + "." + str(amostra) + " capturada com sucesso!")
            amostra += 1
           

    cv2.imshow("Face", imagem) #cria a janela
    #cv2.waitKey(1)# Espera 1ms para processar a imagem
    if (amostra >= numeroAmostras + 1):
           break #para o laço

print("Faces capturadas com sucesso!")
camera.release()
cv2.destroyAllWindows ()

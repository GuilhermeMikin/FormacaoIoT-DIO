import cv2

# LBPH - Local Binary Patterns Histograms

detectorFace = cv2.CascadeClassifier("classificadores\\haarcascade-frontalface-default.xml")
# LBPHFaceRecognizer::create(int radius = 1,
#                            int neighbors = 8,
#                            int grid_x = 8,
#                            int grid_y = 8,
#                            double	threshold = DBL_MAX)
reconhecedor = cv2.face.LBPHFaceRecognizer_create(threshold=2)
reconhecedor.read("classificadorLBPH.yml")

larguraFace = 100
alturaFace = 100

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

camera = cv2.VideoCapture(0) # Captura o video da câmera

while (True):
    conectado, imagem = camera.read() # Captura a imagem da câmera
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # Transforma a imagem em cinza
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(30,30))

    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (larguraFace, alturaFace))
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0,0,255), 2)
        id, confianca = reconhecedor.predict(imagemFace)

        nome = ""

        if id == 1:
            nome = 'Carlos'
        elif id == 2:
            nome = 'Amanda'
        elif id == 3:
            nome = 'Luana'

        # cv2.putText(imagem, nome, (x0, y0 + altura do retângulo + tamanho da fonte mais espaço), fonte, tamanho da fonte, (R,G,B))
        cv2.putText(imagem, nome, (x, y + a + 35), font, 2, (0, 0, 255))
        cv2.putText(imagem, str(confianca), (x, y + a + 55), font, 1, (0, 0, 255))


    cv2.imshow("Face", imagem) # Mostra a imagem identificada mais o valor da confiança
    if cv2.waitKey(1) == ord('q'):  # Pressionar a tecla < q > para sair
        break

camera.release()
cv2.destroyAllWindows()
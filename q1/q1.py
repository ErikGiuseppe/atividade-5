import cv2
import numpy as np

cap = cv2.VideoCapture("q1B.mp4")

def detectar_formas(imagem):
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    cinza = cv2.GaussianBlur(cinza, (5, 5), 0)
    bordas = cv2.Canny(cinza, 50, 150)
    contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    formas = []
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > 100:  
            formas.append((contorno, area))
    
    return formas

def desenhar_maior_contorno(imagem, formas):
    maior_area = 0
    forma_maior = None

    for contorno, area in formas:
        perimetro = cv2.arcLength(contorno, True)
        aproximado = cv2.approxPolyDP(contorno, 0.04 * perimetro, True)


        if area > maior_area:
            maior_area = area
            forma_maior = contorno

    return forma_maior

def calcular_area_sobreposicao(contorno1, contorno2):
    ret, img = cv2.threshold(cv2.drawContours(np.zeros((480, 640), np.uint8), [contorno1], -1, 255, cv2.FILLED), 127, 255, 0)
    ret, img2 = cv2.threshold(cv2.drawContours(np.zeros((480, 640), np.uint8), [contorno2], -1, 255, cv2.FILLED), 127, 255, 0)
    intersection = cv2.bitwise_and(img, img2)
    return cv2.countNonZero(intersection)
def detectar_colisao(contorno1, contorno2, limiar_area=100):
    area_sobreposicao = calcular_area_sobreposicao(contorno1, contorno2)
    return area_sobreposicao > limiar_area

def verificar_ultrapassagem(forma_maior, outras_formas):
    (x1, y1, w1, h1) = cv2.boundingRect(forma_maior)

    for contorno, _ in outras_formas:
        (x2, y2, w2, h2) = cv2.boundingRect(contorno)
        
        if ( x1 + w1 < x2 or y1 > y2 + h2 or y1 + h1 < y2):
            return True
    return False

while True:
    ret, frame = cap.read()

    if not ret:
        break

    formas = detectar_formas(frame)
    
    forma_maior = desenhar_maior_contorno(frame, formas)

    if forma_maior is not None:
        (x, y, w, h) = cv2.boundingRect(forma_maior)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    colisao_detectada = False  

    for i, (contorno1, _) in enumerate(formas):
            for j, (contorno2, _) in enumerate(formas):
                colidiu = detectar_colisao(contorno1, contorno2)
                if colidiu:
                    colisao_detectada = True
                    cv2.putText(frame, "COLISÃO DETECTADA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if forma_maior is not None and verificar_ultrapassagem(forma_maior, formas):
        cv2.putText(frame, "FORMA ULTRAPASSOU", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Exibindo o resultado
    cv2.imshow("Feed", frame)

    # Espera pela tecla 'ESC' para sair
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Código 27 para ESC
        break

# Finalizando a captura de vídeo
cap.release()
cv2.destroyAllWindows()

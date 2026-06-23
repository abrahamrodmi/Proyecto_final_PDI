import cv2
import numpy as np

im = cv2.imread("imagen_sin_procesar_001.png")
def procesamiento(imagen):

    # Escala de grises
    gs = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Rotacion
    fil = gs.shape[0]
    col = gs.shape[1]

    centro = (col // 2, fil // 2)  # y0,x0
    angulo = 0.0
    escala = 1.5

    M = cv2.getRotationMatrix2D(centro, angulo, escala)  # matriz de rotacion

    rotado = cv2.warpAffine(gs, M, (col, fil))

    # Escalamiento
    img = cv2.resize(
        rotado,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC
    )

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(4, 4)
    )

    contraste = clahe.apply(img)

    # Eliminacion de ruido salt n pepper
    mediana = cv2.medianBlur(contraste, 3)

    # Filtro gaussiano — suavizado
    gauss = cv2.GaussianBlur(mediana, (3, 3), 0)


    # Umbral adaptativo gaussiano
    binaria = cv2.adaptiveThreshold(
        gauss,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        20
    )

    # Cierre morfológico
    kernel_cierre = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    cerrada = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel_cierre, iterations=1)

    # borde
    final = cv2.copyMakeBorder(
        cerrada, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255
    )

    return final

imagen = procesamiento(im)
cv2.imshow('imagen', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()
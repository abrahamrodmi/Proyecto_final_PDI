#PARA IR PROBANDO METODOS EFICIENTES PARA IR AGREGANDO A LA FUNCION DEF PROCESAMIENTO
import cv2
import numpy as np

imo = cv2.imread('opalina.jpg')

def eliminar_componentes_pequenos(binaria, area_min=15):

    invertida = cv2.bitwise_not(binaria)  # texto = blanco para el análisis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        invertida, connectivity=8
    )
    salida = np.full(binaria.shape, 255, dtype=np.uint8)
    for i in range(1, num_labels):  # 0 es el fondo
        if stats[i, cv2.CC_STAT_AREA] >= area_min:
            salida[labels == i] = 0
    return salida


def corregir_inclinacion(binaria):

    coords = np.column_stack(np.where(binaria == 0))  # píxeles de texto
    if len(coords) < 10:
        return binaria  # muy poco texto detectado, no rotar

    angulo = cv2.minAreaRect(coords)[-1]
    if angulo < -45:
        angulo = -(90 + angulo)
    else:
        angulo = -angulo

    (h, w) = binaria.shape
    centro = (w // 2, h // 2)
    matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    rotada = cv2.warpAffine(
        binaria, matriz, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )
    return rotada


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

gs = cv2.cvtColor(imo, cv2.COLOR_BGR2GRAY)
contraste = clahe.apply(gs)
img = cv2.resize(
    contraste,
    None,
    fx=2,
    fy=2,
    interpolation=cv2.INTER_CUBIC
)

mediana = cv2.medianBlur(img, 7)


# Filtro gaussiano
filtro = cv2.GaussianBlur(gs, (3, 3), 0)


# Umbralizacion
binaria = cv2.adaptiveThreshold(
    filtro, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,  251, 10
)

kernel = np.ones((2,2),np.uint8)

binary = cv2.morphologyEx(
    binaria,
    cv2.MORPH_CLOSE,
    kernel
)
kernel_cierre = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
cerrada = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_cierre, iterations=1)


final = cv2.copyMakeBorder(
    cerrada, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255
)



cv2.imshow('image', final)
cv2.waitKey(0)
cv2.destroyAllWindows()

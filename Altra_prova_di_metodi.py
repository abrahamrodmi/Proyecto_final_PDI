import cv2
import numpy as np

import cv2
import numpy as np
imo = cv2.imread('no-hoja-alto.jpg')

# Escala de grises
gs = cv2.cvtColor(imo, cv2.COLOR_BGR2GRAY)

# ---------------------------------------------------
# 1. Resize (primero)
# ---------------------------------------------------
img = cv2.resize(
    gs,
    None,
    fx=2,
    fy=2,
    interpolation=cv2.INTER_CUBIC
)

# ---------------------------------------------------
# 2. CLAHE
# ---------------------------------------------------
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

contraste = clahe.apply(img)

# ---------------------------------------------------
# 3. Eliminación de ruido
# ---------------------------------------------------
denoise = cv2.fastNlMeansDenoising(
    contraste,
    None,
    h=15
)

# Alternativa:
# denoise = cv2.medianBlur(contraste, 3)

# ---------------------------------------------------
# 4. Suavizado ligero
# ---------------------------------------------------
filtro = cv2.GaussianBlur(
    denoise,
    (3, 3),
    0
)

# ---------------------------------------------------
# 5. Binarización adaptativa
# ---------------------------------------------------
binaria = cv2.adaptiveThreshold(
    filtro,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    51,
    10
)

# ---------------------------------------------------
# 6. Apertura (quita ruido)
# ---------------------------------------------------
kernel_open = np.ones((2, 2), np.uint8)

abierta = cv2.morphologyEx(
    binaria,
    cv2.MORPH_OPEN,
    kernel_open,
    iterations=1
)

# ---------------------------------------------------
# 7. Cierre (reconstruye letras)
# ---------------------------------------------------
kernel_close = cv2.getStructuringElement(
    cv2.MORPH_CROSS,
    (3, 3)
)

cerrada = cv2.morphologyEx(
    abierta,
    cv2.MORPH_CLOSE,
    kernel_close,
    iterations=1
)

# ---------------------------------------------------
# 8. Deskew
# ---------------------------------------------------
coords = np.column_stack(np.where(cerrada < 255))

if len(coords) > 0:

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = cerrada.shape[:2]

    M = cv2.getRotationMatrix2D(
        (w // 2, h // 2),
        angle,
        1.0
    )

    deskew = cv2.warpAffine(
        cerrada,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )

else:
    deskew = cerrada

# ---------------------------------------------------
# 9. Borde blanco para Tesseract
# ---------------------------------------------------
final = cv2.copyMakeBorder(
    deskew,
    20,
    20,
    20,
    20,
    cv2.BORDER_CONSTANT,
    value=255
)

cv2.imshow('image', final)
cv2.waitKey(0)
cv2.destroyAllWindows()
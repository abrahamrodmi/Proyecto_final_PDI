# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:44:43 2026

@author: SKU 1103671431
"""

#PROYECTO FINAL DE PDI
import cv2
import numpy as np

#CONFIGURACION DE CAMARA

# Abrir cámara
cam = cv2.VideoCapture(0) 

# Configuración deseada
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cam.set(cv2.CAP_PROP_FPS, 60)

if not cam.isOpened():
    print("No se pudo abrir la cámara")
    exit()

print("Presiona:")
print("  g para Guardar imagen")
print("  q para Salir")

contador = 1

#PREPROCESAMIENTO DE IMAGEN

clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

def procesamiento(imagen):
    
    # Escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Filtro gaussiano — kernel 3×3
    gauss = cv2.GaussianBlur(gris, (3, 3), 0)
    
    #Filtro mediana
    #mediana= cv2.medianBlur(gauss, 3)
    
    #CLAHE — mejora contraste 
    contraste = clahe.apply(gauss)

    #Umbral adaptativo gaussiano
    binaria = cv2.adaptiveThreshold(
        contraste,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=15,
        C=8
    )

    #Closing morfológico
    kernel = np.ones((3, 3), np.uint8)
    final  = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel, iterations=1)

    return final
    
imagen_final = None

#LOOP DE CAPTURA
try:
    while True:
        ret, im = cam.read()

        if not ret:
            print("No se pudo capturar la imagen")
            break
        #Ventana de camara
        cv2.imshow("Webcam", im)
        
        tecla = cv2.waitKey(1) & 0xFF
        
        if tecla == ord('g'):
            imo = im.copy()
            
            if tecla == ord('g'):
                imagen_final = procesamiento(im)

                nombre = f"captura_{contador:03d}.png"
                cv2.imwrite(nombre, imagen_final)
                print(f"Guardada: {nombre}")
                contador += 1

                # Vista previa de la imagen procesada
                cv2.imshow("Resultado — pipeline caracteres", imagen_final)

            elif tecla == ord('q'):
                break
            
            
            # Guardar imagen
            nombre_archivo = f"captura_{contador}.jpg"
            cv2.imwrite(nombre_archivo, imagen_final)
            print(f"Imagen guardada: {nombre_archivo}")
            contador += 1
            
        # Salir
        elif tecla == ord('q'):
            break
         
finally:
    if imagen_final is not None:
        cv2.imshow("Ultima captura procesada", imagen_final)
        cv2.waitKey(0)

    cam.release()
    cv2.destroyAllWindows()



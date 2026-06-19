# -*- coding: utf-8 -*-

#################### PROYECTO FINAL DE PDI ####################################

#Importaciones
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import pytesseract
from pytesseract import Output
import textwrap
import difflib

#ADAPTAR LA DIRECCIÓN SEGÚN CORRESPONDA
pytesseract.pytesseract.tesseract_cmd = r'D:\SOFTWARE\tesseract\tesseract.exe'

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
print("  g para Guardar")
print("  q para Salir")

contador = 1

################ PREPROCESAMIENTO DE IMAGEN

clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

def procesamiento(imagen):
    
    # Escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Filtro gaussiano — suavizado
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

    #Cierre morfológico
    kernel = np.ones((3, 3), np.uint8)
    final  = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel, iterations=1)

    return final

#####################################
#                 OCR

def ejecutar_ocr(imagen, idioma='spa', psm=6, oem=3, mostrar_detalle=True):
        #reporta texto y confianza.

        custom_config = f'--oem {oem} --psm {psm} -l {idioma}'

        data = pytesseract.image_to_data(imagen, config=custom_config, output_type=Output.DICT)

        n_boxes = len(data['text'])
        palabras_detectadas = []

        for i in range(n_boxes):
            texto = data['text'][i].strip()
            conf = int(data['conf'][i])

            if texto and conf > 0:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                palabras_detectadas.append(texto)

                if mostrar_detalle:
                    print(f"Texto: '{texto}' | Confianza: {conf}% | Posición: ({x},{y},{w},{h})")

        confidencias_validas = [int(c) for c in data['conf'] if int(c) > 0]
        promedio = None
        if confidencias_validas:
            promedio = sum(confidencias_validas) / len(confidencias_validas)
            print(f"Confianza promedio: {promedio:.2f}%")
        else:
            print("No se detectó texto con confianza válida.")

        return {
            'data': data,
            'confianza_promedio': promedio,
            'texto_completo': ' '.join(palabras_detectadas)
        }
    
#imagen_final = None


def procesar_y_reconocer(imagen, idioma='spa', psm=6, oem=3, mostrar_detalle=True):
    """
    Flujo completo: preprocesamiento + OCR en una sola llamada.
            'imagen_procesada': resultado del preprocesamiento
            'texto_completo': texto detectado
            'confianza_promedio': confianza promedio del OCR
            'data': diccionario crudo de pytesseract
    """

    imagen_final = procesamiento(imagen)
    resultado_ocr = ejecutar_ocr(imagen_final, idioma=idioma, psm=psm, oem=oem, mostrar_detalle=mostrar_detalle)

    return {
        'imagen_procesada': imagen_final,
        'texto_completo': resultado_ocr['texto_completo'],
        'confianza_promedio': resultado_ocr['confianza_promedio'],
        'data': resultado_ocr['data']
    }

def comparar_ocr(imagen_cruda, idioma='spa', psm=6, oem=3):
    """
    Ejecuta OCR tanto en la imagen sin procesar como en su versión
    preprocesada, y compara resultados de eficiencia.
    """
    # OCR sobre imagen SIN procesar (color/cruda, tal cual la entrega la webcam)
    print("=== OCR sobre imagen SIN procesar ===")
    t0 = time.time()
    resultado_cruda = ejecutar_ocr(imagen_cruda, idioma=idioma, psm=psm, oem=oem, mostrar_detalle=False)
    tiempo_cruda = time.time() - t0

    # Preprocesamiento + OCR
    print("\n=== OCR sobre imagen PROCESADA ===")
    imagen_procesada = procesamiento(imagen_cruda)
    t0 = time.time()
    resultado_procesada = ejecutar_ocr(imagen_procesada, idioma=idioma, psm=psm, oem=oem, mostrar_detalle=False)
    tiempo_procesada = time.time() - t0

    n_palabras_cruda = len(resultado_cruda['texto_completo'].split())
    n_palabras_procesada = len(resultado_procesada['texto_completo'].split())

    comparacion = {
        'Sin procesar': {
            'confianza_promedio': resultado_cruda['confianza_promedio'] or 0,
            'palabras_detectadas': n_palabras_cruda,
            'tiempo_ocr': round(tiempo_cruda, 3),
            'texto': resultado_cruda['texto_completo']
        },
        'Procesada': {
            'confianza_promedio': resultado_procesada['confianza_promedio'] or 0,
            'palabras_detectadas': n_palabras_procesada,
            'tiempo_ocr': round(tiempo_procesada, 3),
            'texto': resultado_procesada['texto_completo']
        }
    }

    return comparacion, imagen_procesada


def mostrar_comparacion(comparacion):
    """
    Muestra tabla comparativa y gráfica de barras con los resultados.
    """
    df = pd.DataFrame(comparacion).T
    df_mostrar = df[['confianza_promedio', 'palabras_detectadas', 'tiempo_ocr']]
    df_mostrar.columns = ['Confianza promedio (%)', 'Palabras detectadas', 'Tiempo OCR (s)']

    print("\n=== TABLA COMPARATIVA ===")
    print(df_mostrar.to_string())

    categorias = list(comparacion.keys())
    confianzas = [comparacion[c]['confianza_promedio'] for c in categorias]
    palabras = [comparacion[c]['palabras_detectadas'] for c in categorias]
    colores = ['#d9534f', '#5cb85c']  # rojo = sin procesar, verde = procesada

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(categorias, confianzas, color=colores)
    axes[0].set_title('Confianza promedio del OCR')
    axes[0].set_ylabel('Confianza (%)')
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(confianzas):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha='center')

    axes[1].bar(categorias, palabras, color=colores)
    axes[1].set_title('Palabras detectadas')
    axes[1].set_ylabel('Cantidad')
    for i, v in enumerate(palabras):
        axes[1].text(i, v + 0.1, str(v), ha='center')

    plt.tight_layout()
    plt.show()

    return df_mostrar

def graficar_texto_detectado(comparacion):
    """
    Muestra el texto detectado por cada versión (cruda vs procesada)
    en paneles lado a lado, además de un porcentaje de similitud entre ambos.
    """
    texto_cruda = comparacion['Sin procesar']['texto']
    texto_procesada = comparacion['Procesada']['texto']

    # Similitud global entre ambos textos (0 a 1)
    similitud = difflib.SequenceMatcher(None, texto_cruda, texto_procesada).ratio() * 100

    # Envolver texto largo en líneas legibles
    lineas_cruda = textwrap.wrap(texto_cruda if texto_cruda else "(sin texto detectado)", width=45)
    lineas_procesada = textwrap.wrap(texto_procesada if texto_procesada else "(sin texto detectado)", width=45)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    ax.set_title(f'Texto detectado por OCR  |  Similitud entre versiones: {similitud:.1f}%',
                 fontsize=12, pad=15)

    # Panel izquierdo: sin procesar
    ax.text(0.02, 0.95, 'SIN PROCESAR', fontsize=11, fontweight='bold',
            color='#d9534f', transform=ax.transAxes, va='top')
    ax.text(0.02, 0.85, '\n'.join(lineas_cruda), fontsize=9.5,
            color='black', transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='#fbeaea', edgecolor='#d9534f', pad=0.6))

    # Panel derecho: procesada
    ax.text(0.52, 0.95, 'PROCESADA', fontsize=11, fontweight='bold',
            color='#5cb85c', transform=ax.transAxes, va='top')
    ax.text(0.52, 0.85, '\n'.join(lineas_procesada), fontsize=9.5,
            color='black', transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='#eaf7ea', edgecolor='#5cb85c', pad=0.6))

    plt.tight_layout()
    plt.show()

#LOOP DE CAPTURA
# ============ LOOP DE CAPTURA ============

cam = cv2.VideoCapture(0)
contador = 0
imagen_final = None

try:
    while True:
        ret, im = cam.read()

        if not ret:
            print("No se pudo capturar la imagen")
            break

        # Ventana de cámara
        cv2.imshow("Webcam", im)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord('g'):
            # comparar_ocr() ya llama a procesamiento() internamente,
            # así que no hace falta invocarla aparte
            comparacion, imagen_final = comparar_ocr(im)

            nombre = f"captura_{contador:03d}.png"
            cv2.imwrite(nombre, imagen_final)
            print(f"Guardada: {nombre}")
            contador += 1

            # Vista previa de la imagen procesada
            cv2.imshow("Resultado — pipeline caracteres", imagen_final)

            # Tabla + gráficas comparativas
            tabla = mostrar_comparacion(comparacion)
            graficar_texto_detectado(comparacion)

        elif tecla == ord('q'):
            break

finally:
    if imagen_final is not None:
        cv2.imshow("Última captura procesada", imagen_final)
        cv2.waitKey(0)

    cam.release()
    cv2.destroyAllWindows()



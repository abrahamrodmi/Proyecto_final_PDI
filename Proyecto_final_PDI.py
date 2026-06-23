# -*- coding: utf-8 -*-

"""PROYECTO FINAL DE PDI """

#IMPORTACIONES
#Para la captura de imagen y procesamiento
import cv2
#Para el registro del tiempo de procesamiento del ocr
import time
#Para la grafica de comparación de grados de confianza y comparación de texto reconocido
import matplotlib.pyplot as plt
import pandas as pd
#biblioteca que vincula el motor de procesamiento de caracteres OCR
import pytesseract
from pytesseract import Output
#Para la exportación del texto reconocido a formato .txt
import textwrap
import difflib
import string
#Para el cálculo automático del CER Y WER
from jiwer import wer, cer

#DIRECCIÓN DONDE SE ENCUENTRA EL MOTOR OCR
pytesseract.pytesseract.tesseract_cmd = r'D:\SOFTWARE\tesseract\tesseract.exe'


"""PREPROCESAMIENTO DE IMAGEN"""

def procesamiento(imagen):

# Conversión al espacio de color de escala de grises
    gs = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Rotación por medio de la matriz de rotación
    fil = gs.shape[0]
    col = gs.shape[1]

    centro = (col // 2, fil // 2)  # y0,x0
    angulo = 180.0
    escala = 1.5

    M = cv2.getRotationMatrix2D(centro, angulo, escala)  # matriz de rotacion

    rotado = cv2.warpAffine(gs, M, (col, fil))

# Escalamiento geométrico con función propia de openCV
    img = cv2.resize(
        rotado,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC
    )

# CLAHE:Instancia del algoritmo Contrast Limited Adaptive Histogram Equalization.
#Mejora el contraste de la imagen sin saturar lo que ya esta iluminado
    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(4, 4)
    )
    contraste = clahe.apply(img)

# Eliminacion de ruido sal y pimienta
    mediana = cv2.medianBlur(contraste, 3)

# Filtro gaussiano: suavizado de la imagen
    gauss = cv2.GaussianBlur(mediana, (3, 3), 0)

# Umbral adaptativo gaussiano: Esta es la parte más crítica para el reconocimiento óptico de caracteres (OCR).
    binaria = cv2.adaptiveThreshold(
        gauss,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        #Aumentar C para recortar más ruido.
        20
    )

# Cierre morfológico: altera la forma geométrica de los contornos detectados.
    #Construye un kernel en forma de cruz
    kernel_cierre = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    #Rellenar pequeños agujeros y conectar áreas que están muy juntas.
    cerrada = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel_cierre, iterations=1)
    # Expande el lienzo de la imagen y le añade un margen.
    final = cv2.copyMakeBorder(
        cerrada, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255
    )
    return final


"""OCR"""

def ejecutar_ocr(imagen, idioma='spa', psm=6, oem=3, mostrar_detalle=True):
        """
        Función central que recibe una imagen y extrae el texto utilizando PyTesseract.
        """
        # - oem (OCR Engine Mode): El valor 3 usa el motor moderno de redes neuronales (LSTM)
        # - psm (Page Segmentation Mode): El valor 6 significa un bloque uniforme de texto
        # - l (Language): Le indica al motor que el texto está en español ('spa')
        custom_config = f'--oem {oem} --psm {psm} -l {idioma}'

        #Extracción de datos crudos:
        data = pytesseract.image_to_data(imagen, config=custom_config, output_type=Output.DICT)

        #Procesamiento de los datos obtenidos:
        n_boxes = len(data['text'])
        palabras_detectadas = []

        # Recorremos cada elemento detectado en la imagen
        for i in range(n_boxes):
            texto = data['text'][i].strip()
            conf = int(data['conf'][i])

            if texto and conf > 0:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                palabras_detectadas.append(texto)

                if mostrar_detalle:
                    print(f"Texto: '{texto}' | Confianza: {conf}% | Posición: ({x},{y},{w},{h})")

        # Cálculo de precisión matemática
        # Cálculo de seguridad promedio de lectura
        confidencias_validas = [int(c) for c in data['conf'] if int(c) > 0]
        promedio = None
        if confidencias_validas:
            promedio = sum(confidencias_validas) / len(confidencias_validas)
            print(f"Confianza promedio: {promedio:.2f}%")
        else:
            print("No se detectó texto con confianza válida.")

        # Se retorna el encapsulamiento de datos
        return {
            'data': data,
            'confianza_promedio': promedio,
            'texto_completo': ' '.join(palabras_detectadas)
        }

def procesar_y_reconocer(imagen, idioma='spa', psm=6, oem=3, mostrar_detalle=True):
    """preprocesamiento y OCR en una sola llamada.
            'imagen_procesada': resultado del preprocesamiento
            'texto_completo': texto detectado
            'confianza_promedio': confianza promedio del OCR
    """

    imagen_final = procesamiento(imagen)
    resultado_ocr = ejecutar_ocr(imagen_final, idioma=idioma, psm=psm, oem=oem, mostrar_detalle=mostrar_detalle)

    # Se muestra el resultado final junto con la imagen limpia
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
    imagen_cruda_rotada = cv2.rotate(imagen_cruda, cv2.ROTATE_180)
    print("=== OCR sobre imagen SIN procesar ===")
    t0 = time.time()
    resultado_cruda = ejecutar_ocr(imagen_cruda_rotada, idioma=idioma, psm=psm, oem=oem, mostrar_detalle=False)
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

    return comparacion, imagen_cruda_rotada, imagen_procesada


def mostrar_comparacion(comparacion):
    """
    Muestra tabla comparativa y gráfica de barras con los resultados.
    """
    # Creación de la Tabla (usando la librería Pandas)
    df = pd.DataFrame(comparacion).T
    df_mostrar = df[['confianza_promedio', 'palabras_detectadas', 'tiempo_ocr']]
    df_mostrar.columns = ['Confianza promedio (%)', 'Palabras detectadas', 'Tiempo OCR (s)']

    print("\n=== TABLA COMPARATIVA ===")
    print(df_mostrar.to_string())

    # Creación de los Gráficos (usando Matplotlib)
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


def normalizar_texto(texto: str) -> str:
    """
    Limpia el texto para una comparación justa.
    Convierte a minúsculas, elimina puntuación y espacios extra.
    """
    # Convertir a minúsculas
    texto = texto.lower()
    # Eliminar signos de puntuación
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    # Eliminar saltos de línea y espacios duplicados
    texto = " ".join(texto.split())

    return texto

def evaluar_ocr(texto_real: str, texto_ocr: str):
    """
    Calcula métricas de OCR aplicando normalización previa.
    """
    # 1. Normalización de ambos textos
    texto_real_limpio = normalizar_texto(texto_real)
    texto_ocr_limpio = normalizar_texto(texto_ocr)

    # 2. Cálculo usando jiwer
    cer_valor = cer(texto_real_limpio, texto_ocr_limpio)
    wer_valor = wer(texto_real_limpio, texto_ocr_limpio)

    # 3. Conversión a porcentaje de precisión
    accuracy_caracteres = max((1 - cer_valor) * 100, 0) # Evita porcentajes negativos
    accuracy_palabras = max((1 - wer_valor) * 100, 0)

    return {
        "CER": round(cer_valor, 4),
        "WER": round(wer_valor, 4),
        "Accuracy_Caracteres": f"{round(accuracy_caracteres, 2)}%",
        "Accuracy_Palabras": f"{round(accuracy_palabras, 2)}%"
    }


"""LOOP DE CAPTURA"""

#CONFIGURACION DE CAMARA

    # Abrir cámara
cam = cv2.VideoCapture(0)

    # Configuración de amplitud de camara
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cam.set(cv2.CAP_PROP_FPS, 60)

if not cam.isOpened():
    print("No se pudo abrir la cámara")
    exit()

print("Presiona:")
print("  g para Guardar")
print("  q para Salir")

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
            # Obtenemos el diccionario y ambas imágenes
            comparacion, imagen_rotada, imagen_procesada = comparar_ocr(im)

            contador += 1
            # --- GENERAR NOMBRES DE ARCHIVO ---
            nombre_cruda = f"imagen_sin_procesar_{contador:03d}.png"
            nombre_procesada = f"imagen_procesada_{contador:03d}.png"

            nombre_txt = f"texto_ocr_{contador:03d}.txt"

            # --- GUARDAR IMÁGENES ---
            cv2.imwrite(nombre_cruda, imagen_rotada)
            cv2.imwrite(nombre_procesada, imagen_procesada)
            print(f"Guardadas: {nombre_cruda} y {nombre_procesada}")

            # --- EXTRAER Y GUARDAR TEXTO EN .TXT ---
            # Se extrae el string del diccionario
            texto_extraido = comparacion['Procesada']['texto']

            # archivo en modo escritura ("w") con soporte para español
            with open(nombre_txt, "w", encoding="utf-8") as archivo:
                archivo.write(texto_extraido)
            print(f"Texto guardado en: {nombre_txt}")

            contador += 1

            # Vista previa de la imagen procesada
            cv2.imshow("Resultado del flujo de caracteres", imagen_procesada)
            cv2.imshow("Captura de la imagen de webcam rotada", imagen_rotada)

            # Tabla + gráficas comparativas
            tabla = mostrar_comparacion(comparacion)
            graficar_texto_detectado(comparacion)

            # Texto base de referencia (Ground Truth).
            ground_truth = """1155a Después de esto, podría seguir una discusión sobre la amistad, pues la amistad es una virtud o algo acompañado
            de virtud y, además, es lo más necesario para la vida. En efecto, sin amigos nadie querría vivir, aunque tuviera todos los otros bienes
            ; incluso los que poseen riquezas, autoridad o poder parece que necesitan sobre todo amigos."""

            # 2. Extraer el texto reconocido
            # Se usa el resultado de la imagen 'Procesada' para la evaluación
            ocr = comparacion['Procesada']['texto']

            # 3. Calcular métricas
            resultado = evaluar_ocr(ground_truth, ocr)

            print("\n=== Resultados de Precisión (CER/WER) ===")
            print(f"CER: {resultado['CER']}")
            print(f"WER: {resultado['WER']}")
            print(f"Precisión Caracteres: {resultado['Accuracy_Caracteres']}")
            print(f"Precisión Palabras: {resultado['Accuracy_Palabras']}\n")

        elif tecla == ord('q'):
            break

finally:

    if imagen_final is not None:
        cv2.imshow("Última captura procesada", imagen_final)
        cv2.waitKey(0)

    cam.release()
    cv2.destroyAllWindows()



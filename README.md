Preprocesamiento de imágenes manuscritas para aumentar la precisión del reconocimiento OCR de texto latino y números arábigos.

Objetivo general:
Desarrollar un sistema de preprocesamiento digital de imágenes manuscritas para elevar en al menos un 60% la precisión de motores OCR existentes en el reconocimiento de texto latino y números arábigos.

Descripción:
El proyecto consiste en un sistema que captura y procesa imágenes de palabras manuscritas y números arábigos sobre papel blanco (con iluminación backlight vía webcam) para aumentar su legibilidad antes de pasar por un motor OCR estándar. En lugar de entrenar un nuevo modelo de IA para el reconocimiento de caracteres, la solución se centra exclusivamente en el procesamiento digital de imágenes (PDI). Se aplicará un flujo técnico que incluye: corrección geométrica y compensación de iluminación, reducción de ruido y mejora de contraste, binarización y segmentación de texto.
Finalmente, el sistema evaluará la efectividad del preprocesamiento comparando el porcentaje de precisión del OCR en tres escenarios: la imagen original, la imagen en escala de grises y la imagen procesada. El propósito es demostrar cuantitativamente cómo el PDI es una etapa crítica y suficiente para corregir capturas imperfectas y maximizar el rendimiento de los lectores de caracteres actuales.

Indicaciones:
Para usar el OCR pytesseract es necesario descargar el motor en https://github.com/UB-Mannheim/tesseract/wiki. Instalar la biblioteca pytesseract e indicar la dirección del OCR.
Una vez reconocido pytesseract, bastará con correr el programa en el IDE de preferencia. Las imagenes capturadas se guardaran automáticamente.


# Detector de Personaes en tiempo real con YOLOv4

Este repositorio contiene un algoritmo para detectar objetos en imágenes y videos utilizando YOLOv4.

## Funcionamiento

El funcionamiento del algoritmo se puede resumir en los siguientes pasos:

1. Cargar los archivos de configuración y pesos pre-entrenados de YOLOv4.
2. Obtener los nombres de las clases.
3. Procesar la imagen o el video:
   - Preprocesar la imagen.
   - Pasar la imagen por la red y obtener las detecciones.
   - Procesar las detecciones y dibujar los cuadros delimitadores.
4. Mostrar el resultado con las detecciones.
5. Repetir los pasos 3 y 4 para cada cuadro del video, en caso de estar procesando un video.



## Archivos

- `yolov4-tiny.cfg`: Archivo de configuración de YOLOv4-Tiny.
- `yolov4-tiny.weights`: Pesos pre-entrenados de YOLOv4-Tiny.
- `coco.names`: Archivo que contiene los nombres de las clases de la base de datos COCO.



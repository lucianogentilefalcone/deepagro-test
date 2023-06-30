import numpy as np
import cv2
import logging

# Configurar el archivo de registro
logging.basicConfig(filename='object_detection.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Cargar los archivos de configuración y pesos pre-entrenados de YOLOv4
net = cv2.dnn.readNetFromDarknet('yolov4-tiny.cfg', 'yolov4-tiny.weights')

# Obtener los nombres de las clases
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Configurar la captura de video
cap = cv2.VideoCapture('PEOPLE WALKING IN AIRPORT.mp4')  # Ruta al archivo de video

while True:
    # Leer el cuadro de video
    ret, frame = cap.read()
    if not ret:
        logging.debug("No se devolvio ningun frame")
        break

    # Preprocesar la imagen
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Pasar la imagen por la red y obtener las detecciones
    net.setInput(blob)


    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    # Procesar las detecciones y dibujar los cuadros delimitadores
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.7 and class_id == 0:  # Clase 0: Persona
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f'{classes[class_ids[i]]}: {confidences[i]:.2f}'
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), font, 0.5, color, 2)

    # Mostrar el cuadro de video con las detecciones
    cv2.imshow('Object Detection', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()

import cv2
import dlib
import numpy as np
import os

import time


basedir = os.path.abspath(os.path.dirname(__file__))

# Rutas
face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
# Cargar el detector de rostros de Haar Cascade
face_cascade = cv2.CascadeClassifier(face_cascade_path)
# Cargar el predictor de landmarks de dlib
predictor_path = os.path.join(basedir, 'shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(predictor_path)
# Cargar el detector de rostros de dlib
detector = dlib.get_frontal_face_detector()

import time
def gen(user_name, frames_to_capture=100):
    video_capture = cv2.VideoCapture(0)
    
    user_dir = os.path.join(basedir, 'capturas', user_name)
    
    frame_count = 0

    while True:
        success, frame = video_capture.read()
        if not success:
            break  # Salir del bucle si no se puede capturar video

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame)
        
        
        for face in faces:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            
            # Dibujar rectángulo alrededor del rostro
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Detectar landmarks
            landmarks = predictor(gray_frame, face)
            
            # Dibujar los landmarks
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 1, (255, 255, 0), -1)

        if frame_count < frames_to_capture:
            frame_path = os.path.join(user_dir, f'{frame_count}.jpg')
            os.makedirs(user_dir, exist_ok=True)
            cv2.imwrite(frame_path, frame)
            time.sleep(0.01)  # Espera opcional para ralentizar la captura
        
        frame_count += 1

        if success:
            flag, encodedImage = cv2.imencode(".jpg", frame)
            if flag:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            else:
                break  # Salir del bucle si no se puede codificar la imagen

        if frame_count >= frames_to_capture:
            break

    video_capture.release()
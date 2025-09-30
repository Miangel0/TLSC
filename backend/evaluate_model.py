import os
import cv2
import json
import numpy as np
from mediapipe.python.solutions.hands import Hands
from keras.models import load_model
from helpers import *
from constants import *

def evaluate_model(src=0):
    # Cargar clases
    with open(WORDS_JSON_PATH, "r", encoding="utf-8") as f:
        word_ids = json.load(f)

    # Cargar modelo
    model = load_model(MODEL_PATH)

    # Cargar scaler
    mean = np.load(os.path.join(MODEL_FOLDER_PATH, "scaler.npy"))
    scale = np.load(os.path.join(MODEL_FOLDER_PATH, "scale.npy"))

    def scale_input(x):
        return (x - mean) / scale

    cap = cv2.VideoCapture(src)
    with Hands() as hands_model:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = mediapipe_detection(frame, hands_model)

            if there_hand(results):
                kp_frame = extract_keypoints(results)  # (63,)
                kp_frame = scale_input(kp_frame)

                # Predicción
                res = model.predict(np.expand_dims(kp_frame, axis=0), verbose=0)[0]

                # Seleccionar la palabra más probable
                best_idx = np.argmax(res)
                word_id = word_ids[best_idx]
                sent = words_text.get(word_id, word_id.upper())

                # === MEJORAS VISUALES ===

                # Barra superior semitransparente
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (640, 70), (0, 0, 0), -1)
                alpha = 0.6
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                # Texto centrado con letra más grande
                text = f"{sent} ({res[best_idx]:.2f})"
                font_scale = 1.2  # más grande
                thickness = 3
                (text_w, text_h), _ = cv2.getTextSize(text, FONT, font_scale, thickness)

                x = (frame.shape[1] - text_w) // 2  # centrado horizontal
                y = 45  # un poco abajo de la barra

                # sombra
                cv2.putText(frame, text, (x+2, y+2), FONT, font_scale, (0, 0, 0), thickness+1, cv2.LINE_AA)
                # texto principal
                cv2.putText(frame, text, (x, y), FONT, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                # Barra de probabilidad debajo del texto
                bar_width, bar_height = 300, 20
                bar_x = (frame.shape[1] - bar_width) // 2  # centrado
                bar_y = 60

                prob = res[best_idx]
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + int(bar_width * prob), bar_y + bar_height),
                              (0, 255, 0), -1)

                # Dibujar keypoints en la mano
                draw_keypoints(frame, results)

            # Codificar frame como JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Enviar frame al navegador
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

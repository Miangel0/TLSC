import os
import cv2
import json
import numpy as np
from mediapipe.python.solutions.hands import Hands
from keras.models import load_model
from helpers import *
from constants import *

def evaluate_model(src=None):
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

    with Hands() as hands_model:
        video = cv2.VideoCapture(src or 0)
        sentence = []

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            results = mediapipe_detection(frame, hands_model)

            if there_hand(results):
                kp_frame = extract_keypoints(results)  # (63,)
                kp_frame = scale_input(kp_frame)

                # Predicción
                res = model.predict(np.expand_dims(kp_frame, axis=0), verbose=0)[0]

                # Imprimir todas las probabilidades en consola
                probs = {word_ids[i]: float(res[i]) for i in range(len(word_ids))}
                """ print("Probabilidades:", probs) """

                # Seleccionar la letra más probable
                best_idx = np.argmax(res)
                word_id = word_ids[best_idx]
                sent = words_text.get(word_id, word_id.upper())
                sentence = [sent]

                # Mostrar en pantalla
                cv2.rectangle(frame, (0, 0), (640, 35), (245, 117, 16), -1)
                cv2.putText(frame, f"{sent} ({res[best_idx]:.2f})", FONT_POS, FONT, FONT_SIZE, (255, 255, 255))
                draw_keypoints(frame, results)

            cv2.imshow('Traductor LSC', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()
        return sentence

if __name__ == "__main__":
    evaluate_model()

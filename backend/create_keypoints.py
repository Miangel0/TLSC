import os
import cv2
import numpy as np
import pandas as pd
from mediapipe.python.solutions.hands import Hands
from constants import KEYPOINTS_PATH, FRAME_ACTIONS_PATH
from helpers import get_keypoints, create_folder, mediapipe_detection  # get_keypoints usa mediapipe_detection y extract_keypoints

def create_keypoints(word, words_path, hdf_path):
    """
    Crea un .h5 con los keypoints (vectores de 63 valores) para la clase `word`.
    words_path: carpeta que contiene subcarpetas por palabra (ej. frame_actions/a)
    hdf_path: ruta destino para guardar KEYPOINTS_PATH/<word>.h5
    """
    frames_dir = os.path.join(words_path, word)
    if not os.path.exists(frames_dir):
        print(f"⚠️  Carpeta no encontrada: {frames_dir} -> se omite '{word}'")
        return

    image_files = sorted([
        f for f in os.listdir(frames_dir)
        if os.path.isfile(os.path.join(frames_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if len(image_files) == 0:
        print(f"⚠️  No hay imágenes en: {frames_dir}")
        return

    keypoints_list = []

    # Usamos Hands en modo imagen (static_image_mode=True) porque procesamos fotos
    with Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        for i, fname in enumerate(image_files, start=1):
            img_path = os.path.join(frames_dir, fname)
            # get_keypoints espera (model, img_path) y devuelve vector (63,)
            kp = get_keypoints(hands, img_path)
            keypoints_list.append(kp)
            print(f"{word}: {i}/{len(image_files)}", end="\r")

    # Guardar en HDF5 como una columna 'keypoints' (cada fila = vector 63)
    os.makedirs(os.path.dirname(hdf_path) or KEYPOINTS_PATH, exist_ok=True)
    df = pd.DataFrame({'keypoints': keypoints_list})
    df.to_hdf(hdf_path, key='data', mode='w')
    print(f"\n✅ Keypoints creados para '{word}' -> {len(keypoints_list)} muestras -> {hdf_path}")


if __name__ == "__main__":
    # Asegura carpeta de salida
    create_folder(KEYPOINTS_PATH)

    # Lista de palabras (carpetas) en frame_actions
    word_ids = [w for w in os.listdir(os.path.join(os.getcwd(), FRAME_ACTIONS_PATH))]
    for word in word_ids:
        hdf_path = os.path.join(KEYPOINTS_PATH, f"{word}.h5")
        create_keypoints(word, FRAME_ACTIONS_PATH, hdf_path)

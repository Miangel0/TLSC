import json
import os
import cv2
import numpy as np
import pandas as pd
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec
from mediapipe.python.solutions.hands import HAND_CONNECTIONS
from typing import NamedTuple
from constants import *

# GENERAL
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    return results

def create_folder(path):
    """Crear carpeta si no existe"""
    if not os.path.exists(path):
        os.makedirs(path)

def there_hand(results: NamedTuple) -> bool:
    """Verifica si se detectó al menos una mano"""
    return bool(results.multi_hand_landmarks)

def get_word_ids(path):
    with open(path, 'r') as json_file:
        data = json.load(json_file)
        return data.get('word_ids')

# CAPTURE SAMPLES
def draw_keypoints(image, results, mode="hands"):
    """Dibuja los keypoints de la mano"""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            draw_landmarks(
                image,
                hand_landmarks,
                HAND_CONNECTIONS,
                DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
            )

def save_frames(frames, output_folder):
    for num_frame, frame in enumerate(frames):
        frame_path = os.path.join(output_folder, f"{num_frame + 1}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA))

# CREATE KEYPOINTS
def extract_keypoints(results):
    """Extrae keypoints de una sola mano"""
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # solo la primera mano
        keypoints = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
        return np.array(keypoints).flatten()  # 63 valores
    else:
        return np.zeros(63)  # 21 puntos * 3 coords

def get_keypoints(model, img_path):
    """Obtiene keypoints de una imagen"""
    frame = cv2.imread(img_path)
    results = mediapipe_detection(frame, model)
    kp_frame = extract_keypoints(results)
    return kp_frame

def insert_keypoints_sequence(df, n_sample:int, kp_seq):
    """Inserta los keypoints de una muestra al DataFrame"""
    for frame, keypoints in enumerate(kp_seq):
        data = {'sample': n_sample, 'frame': frame + 1, 'keypoints': [keypoints]}
        df_keypoints = pd.DataFrame(data)
        df = pd.concat([df, df_keypoints])
    return df

# TRAINING MODEL
def get_sequences_and_labels(word_ids):
    """Carga todos los keypoints de cada clase desde los .h5"""
    sequences, labels = [], []
    
    for idx, word in enumerate(word_ids):
        hdf_path = os.path.join(KEYPOINTS_PATH, f"{word}.h5")
        if not os.path.exists(hdf_path):
            print(f"⚠️ No se encontró {hdf_path}, se omite esta clase")
            continue
        
        data = pd.read_hdf(hdf_path, key="data")
        
        # 'keypoints' se guardó como vector numpy (63,)
        for kp in data["keypoints"]:
            sequences.append(np.array(kp, dtype=np.float32))
            labels.append(idx)

    return sequences, labels
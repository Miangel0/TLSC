import os
import json
import numpy as np
from model import get_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from helpers import get_word_ids, get_sequences_and_labels
from constants import *

def training_model(model_path, epochs=200):
    # Obtener lista de clases (letras A-Y)
    word_ids = get_word_ids(WORDS_JSON_PATH)  # ['a', 'b', 'c', ...]

    # Cargar datos desde los keypoints
    sequences, labels = get_sequences_and_labels(word_ids)
    X = np.array(sequences, dtype=np.float32)  # (N, 63)
    y = to_categorical(labels, num_classes=len(word_ids)).astype(int)

    # Normalización
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Guardar scaler para usar en evaluate_model
    np.save(os.path.join(MODEL_FOLDER_PATH, "scaler.npy"), scaler.mean_)
    np.save(os.path.join(MODEL_FOLDER_PATH, "scale.npy"), scaler.scale_)

    # Guardar las clases (word_ids)
    with open(WORDS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(word_ids, f, ensure_ascii=False, indent=2)

    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # Crear modelo
    model = get_model(len(word_ids))

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1)

    # Entrenamiento
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    model.summary()
    model.save(model_path)

if __name__ == "__main__":
    training_model(MODEL_PATH)

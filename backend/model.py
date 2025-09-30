from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from constants import LENGTH_KEYPOINTS

def get_model(output_length: int):
    model = Sequential()
    
    # Capa de entrada
    model.add(Dense(128, activation='relu', input_shape=(LENGTH_KEYPOINTS,), kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    
    # Capas ocultas
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    
    # Capa de salida
    model.add(Dense(output_length, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

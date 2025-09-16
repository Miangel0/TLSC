import numpy as np
import cv2
from evaluate_model import *
from keras.models import load_model
from helpers import *
from constants import *
from mediapipe.python.solutions.holistic import Holistic


model = load_model(MODEL_PATH)
word_ids = get_word_ids(WORDS_JSON_PATH)
sentence = []
kp_seq = []
recording = False
count_frame = 0
fix_frames = 0
margin_frame=1
delay_frames=3
threshold=0.8

def evaluate_model(src=None, threshold=0.8, margin_frame=1, delay_frames=3):
    kp_seq, sentence = [], []
    word_ids = get_word_ids(WORDS_JSON_PATH)
    model = load_model(MODEL_PATH)
    count_frame = 0
    fix_frames = 0
    recording = False
    
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(src or 0, cv2.CAP_DSHOW)
        while video.isOpened():
            ret, frame = video.read()
            if not ret: break
            results = mediapipe_detection(frame, holistic_model)
            # TODO: colocar un máximo de frames para cada seña,
            # es decir, que traduzca incluso cuando hay mano si se llega a ese máximo.
            if there_hand(results) or recording:
                recording = False
                count_frame += 1
                if count_frame > margin_frame:
                    kp_frame = extract_keypoints(results)
                    kp_seq.append(kp_frame)
            
            else:
                if count_frame >= MIN_LENGTH_FRAMES + margin_frame:
                    fix_frames += 1
                    if fix_frames < delay_frames:
                        recording = True
                        continue
                    kp_seq = kp_seq[: - (margin_frame + delay_frames)]
                    kp_normalized = normalize_keypoints(kp_seq, int(MODEL_FRAMES))
                    res = model.predict(np.expand_dims(kp_normalized, axis=0))[0]
                    
                    print(np.argmax(res), f"({res[np.argmax(res)] * 100:.2f}%)")
                    if res[np.argmax(res)] > threshold:
                        word_id = word_ids[np.argmax(res)].split('-')[0]
                        
                        sent = words_text.get(word_id)
                        sentence.insert(0, sent)
                
                recording = False
                fix_frames = 0
                count_frame = 0
                kp_seq = []
            
            if not src:
                # Mostrar la traducción sobre el video
                cv2.rectangle(frame, (0, 0), (960, 35), (245, 117, 16), -1)
                cv2.putText(frame, ' | '.join(sentence[0:]), FONT_POS, FONT, FONT_SIZE, (255, 255, 255))
                draw_keypoints(frame, results)
                
                # Codificar la imagen en formato JPEG
                _, jpeg = cv2.imencode('.jpg', frame)
                frame_bytes = jpeg.tobytes()

                # Enviar la imagen codificada como un "frame" en la secuencia
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        video.release()
        return sentence
    
if __name__ == "__main__":
    evaluate_model()
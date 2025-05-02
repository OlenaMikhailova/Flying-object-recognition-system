import cv2
import numpy as np
from tensorflow.keras.models import load_model

video_path = 'video/plane2.mp4'
model_path = 'model.h5'
IMG_SIZE = 128
class_names = ['Drone', 'Aeroplane', 'Helicopter', 'Bird']

model = load_model(model_path)

def preprocess_frame(frame):
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Не вдалося відкрити відео.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_frame = preprocess_frame(frame)
    prediction = model.predict(input_frame, verbose=0)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    label = f"{predicted_class} ({confidence*100:.1f}%)"
    cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Flying Object Detection', frame)

    # Вихід при натисканні 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

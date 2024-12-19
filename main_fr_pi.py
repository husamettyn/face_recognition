from picamera2 import Picamera2
import face_recognition
import numpy as np
from scipy.spatial.distance import cosine
from preprocess import preprocess_and_save_encodings, load_encodings
from threading import Thread
import time
import cv2

KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.pkl"

resize_factor = 0.2  # Daha düşük çözünürlük
process_frame_interval = 8  # Her 2. karede tanıma yapılacak

known_face_encodings = []
known_face_names = []

choice = input("Encoding oluşturmak ve kaydetmek ister misiniz? (Evet/Hayır): ").strip().lower()

if choice in ["evet", "e"]:
    known_face_encodings, known_face_names = preprocess_and_save_encodings(KNOWN_FACES_DIR, ENCODINGS_FILE)
else:
    known_face_encodings, known_face_names = load_encodings(ENCODINGS_FILE)

face_locations = []
face_encodings = []
face_names = []
frame_count = 0
fps = 0
last_time = time.time()

# Picamera2 başlatma
picam2 = Picamera2()
picam2_config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(picam2_config)
picam2.start()

def process_frame(rgb_small_frame):
    global face_encodings, face_names
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        confidence = 0.0

        # Cosine benzerliği kullanarak en iyi eşleşmeyi bul
        face_distances = [cosine(face_encoding, known_face_encoding) for known_face_encoding in known_face_encodings]
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            confidence = 1 - face_distances[best_match_index]

        face_names.append((name, confidence))

while True:
    frame = picam2.capture_array()

    frame_count += 1

    # FPS hesaplama
    current_time = time.time()
    fps = 1 / (current_time - last_time)
    last_time = current_time

    # Yüz tespiti her karede yapılacak
    small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)

    # Yüz tanıma sadece belirli karelerde yapılacak
    if frame_count % process_frame_interval == 0:
        Thread(target=process_frame, args=(rgb_small_frame,)).start()

    margin = 20
    for (top, right, bottom, left), (name, confidence) in zip(face_locations, face_names):
        top = int(top / resize_factor) - 2 * margin
        right = int(right / resize_factor) + margin
        bottom = int(bottom / resize_factor) + 2 * margin
        left = int(left / resize_factor) - margin

        # Bounding box çizme
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        label = f"{name} ({confidence:.2f})"
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # FPS ekrana yazdırma
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), font, 1, (0, 255, 0), 2)

    # Video ekranını göster
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()

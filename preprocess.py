from tqdm import tqdm
import os
import face_recognition
import pickle
import cv2

def preprocess_and_save_encodings(KNOWN_FACES_DIR, ENCODINGS_FILE, max_size=(1000, 1000)):
    """Preprocess images to extract encodings and save them to a file, resizing images if they are too large."""
    # Create arrays of known face encodings and their names
    known_face_encodings = []
    known_face_names = []
    
    for name in tqdm(os.listdir(KNOWN_FACES_DIR), desc="Klasörler işleniyor"):
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        if not os.path.isdir(person_dir):
            continue

        for filename in tqdm(os.listdir(person_dir), desc=f"{name} klasörü işleniyor", leave=False):
            filepath = os.path.join(person_dir, filename)
            image = face_recognition.load_image_file(filepath)

            # Check if image dimensions exceed max_size and resize if necessary
            height, width, _ = image.shape
            if width > max_size[0] or height > max_size[1]:
                # Calculate the scale factor based on the maximum allowed size
                scale_factor = min(max_size[0] / width, max_size[1] / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                # Resize the image
                image = cv2.resize(image, (new_width, new_height))

            # Convert the resized image to RGB for face_recognition processing
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get the locations of faces in the image
            location = face_recognition.face_locations(rgb_image)
            encodings = face_recognition.face_encodings(rgb_image, location)

            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)

    # Save encodings and names to a file
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

    print(f"{len(known_face_encodings)} yüz kodlandı ve kaydedildi.")
    return known_face_encodings, known_face_names


def load_encodings(ENCODINGS_FILE):
    """Load encodings and names from a file."""
    global known_face_encodings, known_face_names
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'rb') as f:
            known_face_encodings, known_face_names = pickle.load(f)
        print(f"{len(known_face_encodings)} yüz yüklendi.")
        return known_face_encodings, known_face_names
    else:
        print("Kaydedilmiş encoding bulunamadı. Lütfen önişleme yapın.")
        
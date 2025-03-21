import cv2
import face_recognition
import numpy as np
import os

image_paths = [
    r"path gambar",
    r"path gambar",
    r"path gambar",
]
names = ["a", "b", "c"]

known_face_encodings = []
known_face_names = []

for img_path, name in zip(image_paths, names):
    if not os.path.exists(img_path):
        print(f"[WARNING] Gambar tidak ditemukan: {img_path}")
        continue

    image = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(image)

    if encodings:
        known_face_encodings.append(encodings[0])
        known_face_names.append(name)
    else:
        print(f"[WARNING] Tidak ada wajah yang terdeteksi di: {img_path}")

video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not video_capture.isOpened():
    print("[ERROR] Tidak dapat membuka kamera!")
    exit()

print("[INFO] Cam started. Press q to exit.")

frame_skip = 55  
frame_count = 0
face_locations = []
face_encodings = []
face_names = []

while True:
    ret, frame = video_capture.read()
    
    if not ret:
        print("[ERROR] Tidak dapat mengambil gambar dari kamera!")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame_count % frame_skip == 0:
        face_locations = face_recognition.face_locations(rgb_frame, model="hog") 
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
                name = known_face_names[best_match_index]

            face_names.append(name)

    frame_count += 1

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  
        cv2.putText(frame, name, (left, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2) 

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

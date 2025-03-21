import cv2
import face_recognition
import numpy as np
import os

# Pastikan gambar yang akan digunakan ada
image_paths = [
    r"D:\FA_CYBERSPACE\image\SMT6\facerecogpr\faces\azfa.png",
    r"D:\FA_CYBERSPACE\image\SMT6\facerecogpr\faces\ollie.jpeg",
    r"D:\coolyeah\THINGS\comvis_try\ryu.jpeg"
]
names = ["faa", "oliver", "ryu"]

known_face_encodings = []
known_face_names = []

# Load gambar hanya sekali (menghemat waktu)
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

# Mulai kamera
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Kurangi resolusi untuk performa
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not video_capture.isOpened():
    print("[ERROR] Tidak dapat membuka kamera!")
    exit()

print("[INFO] Cam started. Press q to exit.")

# Variabel untuk skip frame
frame_skip = 55  # Periksa wajah hanya setiap 2 frame
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

    # Hanya lakukan face recognition setiap `frame_skip` frame
    if frame_count % frame_skip == 0:
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")  # "hog" lebih cepat
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

    # Gambar kotak wajah & nama
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  # Kotak merah
        cv2.putText(frame, name, (left, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)  # Nama putih

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan sumber daya
video_capture.release()
cv2.destroyAllWindows()

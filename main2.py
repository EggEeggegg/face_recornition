import cv2
from insightface.app import FaceAnalysis
import numpy as np
import os
from numpy import dot
from numpy.linalg import norm

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])  # ใช้ GPU
app.prepare(ctx_id=0)  # 0 คือ GPU ตัวแรก


rtsp_url = "http://192.168.200.82:8080/video" 
cap = cv2.VideoCapture(rtsp_url)

# โหลดฐานข้อมูลใบหน้าทั้งหมด
known_embeddings = []
known_names = []

known_dir = "known_faces"  # โฟลเดอร์เก็บภาพฐานข้อมูล

for filename in os.listdir(known_dir):
    path = os.path.join(known_dir, filename)
    name = os.path.splitext(filename)[0]
    img = cv2.imread(path)
    if img is None:
        continue
    if img.shape[0] < 300:  # ถ้าสูงน้อยกว่า 300px
        img = cv2.resize(img, None, fx=2, fy=2)
    faces = app.get(img)
    if faces:
        known_embeddings.append(faces[0].embedding)
        known_names.append(name)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    faces = app.get(frame)  # frame ต้องเป็น BGR (เช่นจาก OpenCV)   
    
    # วาดกรอบใบหน้า
    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # สีเขียว หนา 2 px

        # Optional: แสดงอายุและเพศ
        # label = f"{'M' if face.gender == 1 else 'F'}, {face.age}"
        # cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
        vec1 = face.embedding
        # เปรียบเทียบกับทุกคนในฐานข้อมูล
        best_score = -1
        best_name = "Unknown"
        for i, known_vec in enumerate(known_embeddings):
            similarity = dot(known_vec, vec1) / (norm(known_vec) * norm(vec1))
            if similarity > best_score:
                best_score = similarity
                best_name = known_names[i]


        # print("Best Similarity:", best_score)

        if best_score < 0.4:
            best_name = "Unknown"


        # วาดกรอบและชื่อ
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{best_name} ({best_score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

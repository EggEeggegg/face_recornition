import cv2
import os
import insightface
from insightface.app import FaceAnalysis
from numpy import dot
from numpy.linalg import norm

# สร้าง FaceAnalysis object โดยใช้ GPU
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])  # ใช้ GPU
app.prepare(ctx_id=0)  # 0 คือ GPU ตัวแรก

# โหลดฐานข้อมูลใบหน้าทั้งหมด
known_embeddings = []
known_names = []

known_dir = "known_faces"  # โฟลเดอร์เก็บภาพฐานข้อมูล
for filename in os.listdir(known_dir):
    path = os.path.join(known_dir, filename)
    name = os.path.splitext(filename)[0]
    img = cv2.imread(path)
    faces = app.get(img)
    if faces:
        known_embeddings.append(faces[0].embedding)
        known_names.append(name)

def face_detect(img2):
    # โหลดภาพที่ต้องการเปรียบเทียบ

    # ตรวจจับใบหน้าในแต่ละภาพ
    # faces1 = app.get(img1)
    faces2 = app.get(img2)

    # ตรวจสอบว่ามีใบหน้าในทั้งสองภาพ
    if not faces2:
        print("ไม่พบใบหน้าในภาพที่ตรวจสอบ")
        found_face = False
        return found_face, ""
    
    vec2 = faces2[0].embedding

    # เปรียบเทียบกับทุกคนในฐานข้อมูล
    best_score = -1
    best_name = "Unknown"
    for i, known_vec in enumerate(known_embeddings):
        similarity = dot(known_vec, vec2) / (norm(known_vec) * norm(vec2))
        if similarity > best_score:
            best_score = similarity
            best_name = known_names[i]


    print("Best Similarity:", best_score)
    if best_score > 0.35:  # กำหนด threshold ตามคุณภาพโมเดลและภาพ
        print("MATCH ->", best_name)
        return True, best_name
    else:
        print("NOT MATCH")
        return True, "NOT MATCH"

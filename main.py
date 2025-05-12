import cv2
import os
import shutil
import threading
from model import predict
from face_detect import face_detect

def run_face_detect(img):
    global is_face_detecting, text_result
    is_face_detecting = True
    # img = cv2.imread("captures/person.jpg")  # โหลดใหม่ เพราะ save ใหม่เรื่อย ๆ
    found_face, check = face_detect(img)
    text_result = f"Face is {check}" if found_face else "Not Found"
    is_face_detecting = False

# ใส่ URL RTSP ที่ได้จากมือถือหรือกล้อง IP
rtsp_url = "http://192.168.200.82:8080/video"  # เช่น rtsp://192.168.1.100:8554/live
input_size_show = (1280,720)
input_size = (640, 640) 


save_dir = "captures"
# เคลียโฟเดอร์เก่า
if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
# สร้างโฟเดอร์ใหม่
os.makedirs(save_dir, exist_ok=True)

# เปิดสตรีม RTSP
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("ไม่สามารถเชื่อมต่อกล้อง RTSP ได้")
    exit()

save_threshold = 20
detect_counter = 0
frame_count = 0  # เพิ่มตัวแปรนับจำนวนเฟรม
results = []     # เก็บผลลัพธ์ล่าสุดเอาไว้ใช้ซ้ำ
found_person = False
valid_box = None
target_width, target_height = input_size
start_face_detect = False
is_face_detecting = False
text_result = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถอ่านเฟรมจากกล้องได้")
        break

    frame_count += 1
    
    resized_show = cv2.resize(frame, input_size_show)

    scale_x = input_size_show[0] / target_width
    scale_y = input_size_show[1] / target_height
    
    # ทำ predict ทุก 4 เฟรมเท่านั้น
    if frame_count % 4 == 0 and not is_face_detecting:

        results = predict(frame, input_size)
        
        for result in results:
            for box in result.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # แปลงพิกัดกลับ
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                conf = float(box.conf[0])
                if conf > 0.7:
                    found_person = True
                    valid_box = (x1, y1, x2, y2, conf)
                    cv2.rectangle(resized_show, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(resized_show, f"Person {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # นับเฉพาะเมื่อเจอคน
            if found_person:
                detect_counter += 1
            else:
                detect_counter = 0

            if detect_counter >= save_threshold:
                detect_counter = 0
                x1, y1, x2, y2, conf = valid_box
                cropped = resized_show[y1:y2, x1:x2]
                # ตรวจว่ากรอบไม่เล็กเกินไป (กันบั๊ก)
                if cropped.size > 0 and cropped.shape[0] > 20 and cropped.shape[1] > 20:
                    # timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"{save_dir}/person.jpg"
                    # cv2.imwrite(filename, cropped)
                    # print(f"Save Image: {filename}")
                    if not is_face_detecting:
                        threading.Thread(target=run_face_detect(cropped)).start()
                    


    cv2.putText(resized_show, text_result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2, cv2.LINE_AA)

    # แสดงผล
    cv2.imshow("YOLOv8 Detection", resized_show)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
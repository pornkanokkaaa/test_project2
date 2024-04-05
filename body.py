import mediapipe as mp
import cv2



#### แบ่งส่วนร่างกาย #####
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def divide_body_parts(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Unable to read the image. Please check the image path.")

        image_height, image_width, _ = image.shape

        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks is None:
                return None

            # หาค่า y ของส่วนต่าง ๆ ของร่างกาย
            head_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y 
            chest_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y 
            mid_body_y = ((results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y +
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y))/ 2 
            hips_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y 
            print(head_y)
            print(chest_y)
            print(mid_body_y)
            print(hips_y)
            # วาดจุดที่ใช้ในการคำนวณ
            for landmark in [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP]:
                landmark_point = results.pose_landmarks.landmark[landmark]
                landmark_x = int(landmark_point.x * image_width)
                landmark_y = int(landmark_point.y * image_height)
                cv2.circle(image, (landmark_x, landmark_y), 5, (255, 0, 0), -1)

            # วาดเส้นตำแหน่งของแต่ละส่วนบนภาพ
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # วาดเส้นแบ่งส่วนของร่างกาย
            cv2.line(annotated_image, (0, head_y), (image_width, head_y), (255, 0, 0), 2)        # หัว
            cv2.line(annotated_image, (0, chest_y), (image_width, chest_y), (0, 255, 0), 2)      # หน้าอกบน
            cv2.line(annotated_image, (0, mid_body_y), (image_width, mid_body_y), (0, 0, 255), 2)  # กลางตัว
            cv2.line(annotated_image, (0, hips_y), (image_width, hips_y), (255, 255, 0), 2)      # ท้อง

            # ปรับขนาดรูปที่แสดงในหน้าต่างเป็น 800x800
            resized_image = cv2.resize(annotated_image, (800, 800))

            cv2.imshow('Body Parts Division', resized_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return head_y, chest_y, mid_body_y, hips_y
    except Exception as e:
        print("Error:", e)
        return None

# เรียกใช้งานฟังก์ชัน
image_path = "./data/1/1 - Copy (2).jpg"
body_parts = divide_body_parts(image_path)
if body_parts is not None:
    print("Head Y:", body_parts[0])
    print("Chest Y:", body_parts[1])
    print("Mid Body Y:", body_parts[2])
    print("Hips Y:", body_parts[3])

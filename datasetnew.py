import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(
    static_image_mode=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle 

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):

        data_aux = []
        x_left = []
        y_left = []
        x_right = []
        y_right = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        results = holistic.process(img_rgb)
        
        height, width, _ = img.shape
        
        if results.left_hand_landmarks is not None:
            left_hand_landmarks = results.left_hand_landmarks.landmark

            # Find the minimum x and y values
            min_x = min([landmark.x for landmark in left_hand_landmarks])
            min_y = min([landmark.y for landmark in left_hand_landmarks])

            for landmark in left_hand_landmarks:
                x_left.append(landmark.x)
                y_left.append(landmark.y)

                # Subtract the minimum x and y values from each landmark coordinate
                data_aux.append(landmark.x - min_x)
                data_aux.append(landmark.y - min_y)
        else:
            # If no left hand landmarks are detected, fill the data_aux list with 0.999
            data_aux.extend([1] * 42)

        if results.right_hand_landmarks is not None:
            right_hand_landmarks = results.right_hand_landmarks.landmark

            # ค้นหาค่า x และ y ที่ต่ำที่สุด
            min_x = min([landmark.x for landmark in right_hand_landmarks])
            min_y = min([landmark.y for landmark in right_hand_landmarks])

            for landmark in right_hand_landmarks:
                x_right.append(landmark.x)
                y_right.append(landmark.y)

                # ลบค่า x และ y ที่ต่ำที่สุดออกจากทุกๆ จุดตำแหน่ง
                data_aux.append(landmark.x - min_x)
                data_aux.append(landmark.y - min_y)
        else:
            # หากไม่พบจุดที่มือขวาตรวจจับได้ ให้เติมรายการ data_aux ด้วย 0.999
            data_aux.extend([1] * 42)

            ###### มุมบน มุมแขน #######
        if results.pose_landmarks is not None:
                    pose_landmarks = results.pose_landmarks.landmark
                    
                    # left 
                    shoulder_left = [pose_landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x, pose_landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow_left = [pose_landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x, pose_landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist_left = [pose_landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x, pose_landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]

                    # right
                    shoulder_right = [pose_landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x, pose_landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow_right = [pose_landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x, pose_landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist_right = [pose_landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x, pose_landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]

                    left_arm_angle1 = calculate_angle(shoulder_left, elbow_left, wrist_left)
                    right_arm_angle1 = calculate_angle(shoulder_right, elbow_right, wrist_right)

                    under_angle = [left_arm_angle1, right_arm_angle1]

        else:
            data_aux.extend([1] * 2)

        data_aux.extend(under_angle)
        ######## มุมล่าง มุมลำตัว ######## 
        if results.pose_landmarks is not None:
                    pose_landmarks = results.pose_landmarks.landmark
                    # left 
                    shoulder_left = [pose_landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].x, pose_landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].y]
                    elbow_left = [pose_landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x, pose_landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
                    wrist_left = [pose_landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x, pose_landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]

                    # right
                    shoulder_right = [pose_landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].x, pose_landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].y]
                    elbow_right = [pose_landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x, pose_landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
                    wrist_right = [pose_landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x, pose_landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]

                    left_arm_angle2 = calculate_angle(shoulder_left, elbow_left, wrist_left)
                    right_arm_angle2 = calculate_angle(shoulder_right, elbow_right, wrist_right)

                    bottom_angle = [left_arm_angle2, right_arm_angle2]
                    data_aux.extend(bottom_angle)

                    labels.append(dir_)
                    data.append(data_aux)

        else:
            data_aux.extend([1] * 2)
    


print(len(data))
print(len(labels))

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
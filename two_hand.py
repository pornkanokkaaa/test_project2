


####### อันนี้ที่ใช้เก็บข้อมูลสองมือจริงๆ #####
import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd

mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

pose = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

DATA_DIR = './data'

data_left = []
data_right = []
data_ud_angle = []
data_bt_angle = []
labels = []

def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle =  360 - angle
    return angle

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):

        data_left_aux = [] 
        data_right_aux = [] 

        x_left = []
        y_left = []
        x_right = []
        y_right = []
        x_pose = []
        y_pose = []

        img1 = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img = cv2.resize(img1, (800, 800))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        
        results_holistic = holistic.process(img_rgb)
        results_pose = pose.process(img)

        ########### มือซ้าย ##########
        if results_holistic.left_hand_landmarks is not None:
            left_hand_landmarks = results_holistic.left_hand_landmarks  

            for i in range(len(left_hand_landmarks.landmark)):
                x = left_hand_landmarks.landmark[i].x
                y = left_hand_landmarks.landmark[i].y

                x_left.append(x)
                y_left.append(y)

                data_left_aux.append(x - min(x_left))
                data_left_aux.append(y - min(y_left))
        else:
            data_left_aux = [0.999] * 42
            # print(data_left_aux)
        data_left.append(data_left_aux)

        ############# มือขวา ##############
        if results_holistic.right_hand_landmarks is not None:
            right_hand_landmarks = results_holistic.right_hand_landmarks  

            for i in range(len(right_hand_landmarks.landmark)):
                x = right_hand_landmarks.landmark[i].x
                y = right_hand_landmarks.landmark[i].y

                x_right.append(x)
                y_right.append(y)

                data_right_aux.append(x - min(x_right))
                data_right_aux.append(y - min(y_right))
        else:
            data_right_aux = [0.999] * 42
            # print(data_right_aux)
        data_right.append(data_right_aux)


         ###### มุมบน #######
        if results_pose.pose_landmarks is not None:
            pose_landmarks = results_pose.pose_landmarks.landmark
            
            # left 
            shoulder_left = [pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_left = [pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_left = [pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # right
            shoulder_right = [pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_right = [pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_right = [pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            left_arm_angle1 = calculate_angle(shoulder_left, elbow_left, wrist_left)
            right_arm_angle1 = calculate_angle(shoulder_right, elbow_right, wrist_right)
            under_angle = [left_arm_angle1,right_arm_angle1]
            # print("under", under_angle)
        data_ud_angle.append(under_angle)

            ######## มุมล่าง ######## 
        if results_pose.pose_landmarks is not None:
            pose_landmarks = results_pose.pose_landmarks.landmark
            
            # left 
            shoulder_left = [pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            elbow_left = [pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            wrist_left = [pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

            # right
            shoulder_right = [pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            elbow_right = [pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            wrist_right = [pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

            left_arm_angle2 = calculate_angle(shoulder_left, elbow_left, wrist_left)
            right_arm_angle2 = calculate_angle(shoulder_right, elbow_right, wrist_right)
            bottom_angle = [left_arm_angle2,right_arm_angle2]
            # print("bottom", bottom_angle)
        data_bt_angle.append(bottom_angle)

        
        labels.append(dir_)



f_left = open('data_left_hand.pickle', 'wb')
pickle.dump({'data': data_left, 'labels': labels}, f_left)
f_left.close()

f_right = open('data_right_hand.pickle', 'wb')
pickle.dump({'data': data_right, 'labels': labels}, f_right)
f_right.close()

f_ud_angle = open('data_ud_angle.pickle', 'wb')
pickle.dump({'data': data_ud_angle, 'labels': labels}, f_ud_angle)
f_ud_angle.close()

f_bt_angle = open('data_bt_angle.pickle', 'wb')
pickle.dump({'data': data_bt_angle, 'labels': labels}, f_bt_angle)
f_bt_angle.close()
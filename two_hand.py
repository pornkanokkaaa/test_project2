


####### อันนี้ที่ใช้เก็บข้อมูลสองมือจริงๆ #####
import os
import pickle
import mediapipe as mp
import cv2
import pandas as pd

mp_holistic = mp.solutions.holistic
num_landmarks = 21
mp_drawing = mp.solutions.drawing_utils


holistic = mp_holistic.Holistic(
    static_image_mode=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

DATA_DIR = './data'

data_left = []
data_right = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):

        data_left_aux = [] 
        data_right_aux = [] 

        x_left = []
        y_left = []
        x_right = []
        y_right = []

        img1 = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img = cv2.resize(img1, (800, 800))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        
        results = holistic.process(img_rgb)

        ########### มือซ้าย ##########
        if results.left_hand_landmarks is not None:
            left_hand_landmarks = results.left_hand_landmarks  

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
        if results.right_hand_landmarks is not None:
            right_hand_landmarks = results.right_hand_landmarks  

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
        labels.append(dir_)



f_left = open('data_left_hand.pickle', 'wb')
pickle.dump({'data': data_left, 'labels': labels}, f_left)
f_left.close()

f_right = open('data_right_hand.pickle', 'wb')
pickle.dump({'data': data_right, 'labels': labels}, f_right)
f_right.close()
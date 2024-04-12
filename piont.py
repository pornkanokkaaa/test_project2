import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
import math

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB 
    image.flags.writeable = False                  # Image is no longer writeable เขียนไม่ได้
    results = model.process(image)                 # Make prediction คาดการณ์ ตรวจจับได
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
# เมื่อไม่เจอให้มีค่าเป็นศูนย์
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

image = cv2.imread('Phototest/001.png')
image = cv2.resize(image, (600, 600))

point =  holistic.process(image)
height, width, _ = image.shape

#เรียกจุด

if point.pose_landmarks is not None:
    pose_landmarks = point.pose_landmarks.landmark

    # right
    shoulder_right = [pose_landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x, pose_landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
    # left 
    shoulder_left = [pose_landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x, pose_landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]

    #print(shoulder_left)

#มือซ้าย

if point.left_hand_landmarks is not None:
    left_hand_landmarks = point.left_hand_landmarks.landmark

    # จุดปลายนิ้วชี้
    tip_x = int(left_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x * width)
    tip_y = int(left_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y * height)
    
    # วาดจุดปลายนิ้วชี้
    cv2.circle(image, (tip_x, tip_y), 5, (0, 255, 0), -1)

    # เฉลี่ยของทุกจุดบนมือซ้าย
    left_sum_x = 0
    left_sum_y = 0

    for landmark in left_hand_landmarks:
        left_sum_x += landmark.x
        left_sum_y += landmark.y

    num_landmarks = len(left_hand_landmarks)
    average_left_x = int(left_sum_x / num_landmarks * width)
    average_left_y = int(left_sum_y / num_landmarks * height)
    
    # แสดงผลภาพจุดนิ้วชี้
    # cv2.imshow('Result', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # วาดจุดเฉลี่ยบนภาพ
    cv2.circle(image, (average_left_x, average_left_y), 5, (0, 255, 0), -1)

    # ผสมภาพต้นฉบับกับภาพที่มีจุดเฉลี่ย
    result_image = cv2.addWeighted(image, 1, image, 0.5, 0)

    # แสดงผลภาพที่มีจุดเฉลี่ย+นิ้วชี้
    # cv2.imshow('Result', result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# มือขวา
    
if point.right_hand_landmarks is not None:
    right_hand_landmarks = point.right_hand_landmarks.landmark

    # จุดปลายนิ้วชี้ของมือขวา
    tip_x = int(right_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x * width)
    tip_y = int(right_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y * height)

    # วาดจุดปลายนิ้วชี้ของมือขวา
    cv2.circle(image, (tip_x, tip_y), 5, (0, 255, 0), -1)

    # คำนวณเฉลี่ยของทุกจุดบนมือขวา
    right_sum_x = 0
    right_sum_y = 0

    for landmark in right_hand_landmarks:
        right_sum_x += landmark.x
        right_sum_y += landmark.y

    num_landmarks = len(right_hand_landmarks)
    average_right_x = int(right_sum_x / num_landmarks * width)
    average_right_y = int(right_sum_y / num_landmarks * height)
    
    # วาดจุดเฉลี่ยของมือขวา
    cv2.circle(image, (average_right_x, average_right_y), 5, (0, 255, 0), -1)

    # ผสมภาพต้นฉบับกับภาพที่มีจุดเฉลี่ยของมือขวา
    result_image = cv2.addWeighted(image, 1, image, 0.5, 0)

    # แสดงผลภาพที่มีการวาดจุดปลายนิ้วชี้และจุดเฉลี่ยของมือขวา
    # cv2.imshow('Result', result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Make detections
    image, results = mediapipe_detection(image, holistic)
    # Draw landmarks
    draw_styled_landmarks(image, results)

# Display image
cv2.imshow('OpenCV Feed', image)
cv2.waitKey(0)
cv2.destroyAllWindows()



        
        

                    





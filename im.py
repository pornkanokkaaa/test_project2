


############# อันนี้เอาไว้วาดเฉยๆ ไม่เกี่ยวอะไร ##########
import cv2
import mediapipe as mp
import os

mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic()

image_path = 'data/1/1.png'

image = cv2.imread(image_path)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = holistic.process(image_rgb)

mp.solutions.drawing_utils.draw_landmarks(image, results.face_landmarks)
mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

cv2.imshow('Holistic Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

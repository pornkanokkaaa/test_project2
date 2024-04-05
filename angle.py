import cv2
import mediapipe as mp
import math
import numpy as np



####### มุม ########
mp_pose = mp.solutions.pose

imgg = cv2.imread("data/2/2 - Copy (2).jpg")
img = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB) 

pose = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

results = pose.process(img)

def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle =  360 - angle
    return angle

# มุมบน 
if results.pose_landmarks is not None:
    pose_landmarks = results.pose_landmarks.landmark
            
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
    print("under", under_angle)

# มุมล่าง 
if results.pose_landmarks is not None:
    pose_landmarks = results.pose_landmarks.landmark
            
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
    print("bottom", bottom_angle)
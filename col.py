import os
import cv2
import mediapipe as mp



############ ยังไม่เสร็จดี ไม่แน่ใจว่าทำอะไรไว้ ##########

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 2
dataset_size = 100

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
        mp_face.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face, \
        mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    for j in range(number_of_classes):
        if not os.path.exists(os.path.join(DATA_DIR, str(j))):
            os.makedirs(os.path.join(DATA_DIR, str(j)))

        print('Collecting data for class {}'.format(j))

        done = False
        while True:
            ret, frame = cap.read()
            cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_frame)
            face_results = face.process(rgb_frame)
            hands_results = hands.process(rgb_frame)

            # ถ้ามีตำแหน่งของตัว
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # ถ้ามีใบหน้า
            if face_results.multi_face_landmarks:
                for landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(frame, landmarks, mp_face.FACEMESH_CONTOURS)

            # ถ้ามีมือ
            if hands_results.multi_hand_landmarks:
                for landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('q'):
                break

        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_frame)
            face_results = face.process(rgb_frame)
            hands_results = hands.process(rgb_frame)

            # ถ้ามีตำแหน่งของตัว
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # ถ้ามีใบหน้า
            if face_results.multi_face_landmarks:
                for landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(frame, landmarks, mp_face.FACEMESH_CONTOURS)

            # ถ้ามีมือ
            if hands_results.multi_hand_landmarks:
                for landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('frame', frame)
            cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
            cv2.waitKey(25)

            counter += 1

cap.release()
cv2.destroyAllWindows()

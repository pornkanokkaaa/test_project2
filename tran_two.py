import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image

font = ImageFont.truetype("./font/THSarabun.ttf", 50)

model_dict = pickle.load(open('./model_alldata.p', 'rb'))
model = model_dict['model']


########### แปลรวมมั้ง ########
cap = cv2.VideoCapture()
cap.open('vdo/fine.mov') 
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

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

def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle =  360 - angle
    return angle


labels_dict = {1: '001', 2: '002', 3: '003', 4:'004', 5:'005', 6:'006',7:'007',8:'008'}
# prev_hand_state = None
all_data = []
data_ = []
massage = ''

while(cap.isOpened()):

    all_data_aux = []
    x_left = []
    y_left = []
    x_right = []
    y_right = []



    ret, frame = cap.read()
    if ret == True :
     H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_holistic = holistic.process(frame_rgb)
    results_pose = pose.process(frame_rgb)

    # # หายตามมือ
    # hand_state = "OPEN" if results.multi_hand_landmarks else "CLOSED"
    # if hand_state != prev_hand_state:
    #     data__ = []
    #     joined_data = ''
    #     data_ = []
    #     massage = ''
    # prev_hand_state = hand_state


    ######### มือซ้าย #########
    if results_holistic.left_hand_landmarks is not None:
        left_hand_landmarks = results_holistic.left_hand_landmarks

        for i in range(len(left_hand_landmarks.landmark)):
            x = left_hand_landmarks.landmark[i].x
            y = left_hand_landmarks.landmark[i].y

            x_left.append(x)
            y_left.append(y)

            all_data_aux.append(x - min(x_left))
            all_data_aux.append(y - min(y_left))
    else :
        all_data_aux.extend([0.999] * 42) 


    ######### มือขวา #########
    if results_holistic.right_hand_landmarks is not None:
        right_hand_landmarks = results_holistic.right_hand_landmarks

        for i in range(len(right_hand_landmarks.landmark)):
            x = right_hand_landmarks.landmark[i].x
            y = right_hand_landmarks.landmark[i].y

            x_right.append(x)
            y_right.append(y)

            all_data_aux.append(x - min(x_right))
            all_data_aux.append(y - min(y_right))
    else:
        all_data_aux.extend([0.999] * 42) 


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
        all_data_aux.extend(under_angle)

     ######## มุมล่าง ######## 
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
        all_data_aux.extend(bottom_angle)

    # all_data.append(all_data_aux)

    prediction = model.predict([np.asarray(all_data_aux)])
    predicted_character = labels_dict[int(prediction[0])]
    data_.append(predicted_character)
    massage = ''.join(data_) 
        # print(massage)
     # print(all_data_aux)
     # 88
    ########## แพ้อาหาร ########
    if '001002' and '002003' in massage:
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.text((70, 5), "แพ้อาหาร", font=font, fill=(255, 255, 255))
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    ########## สบายดี ########
    if '004005' in massage:
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.text((70, 5), "สบายดี", font=font, fill=(255, 255, 255))
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    #     ########## ฉัน ########
    # if '006' in massage:
    #     frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #     draw = ImageDraw.Draw(frame_pil)
    #     draw.text((70, 5), "ฉัน", font=font, fill=(255, 255, 255))
    #     frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            ########## หมอ ########
    # if '006007' and '007008'in massage:
    #     frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #     draw = ImageDraw.Draw(frame_pil)
    #     draw.text((70, 5), "แพทย์", font=font, fill=(255, 255, 255))
    #     frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    
print(model_dict)

cap.release()
cv2.destroyAllWindows()
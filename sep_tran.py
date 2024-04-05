import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image


####### แปลแยก ########

font = ImageFont.truetype("./font/THSarabun.ttf", 50)

model_dict_left = pickle.load(open('./model_left.p', 'rb'))
model_left = model_dict_left['model']

model_dict_right = pickle.load(open('./model_right.p', 'rb'))
model_right = model_dict_right['model']


cap = cv2.VideoCapture()
cap.open('vdo/DDDD.mp4') 
# cap.open('vdo/finee.mp4') 
# cap.open('vdo/DDD.mov') 
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(
    static_image_mode=True,
    min_detection_confidence=0.15,
    min_tracking_confidence=0.15,
)

pose = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.15,
    min_tracking_confidence=0.15,
)

# def calculate_angle(a,b,c):
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)

#     radians = np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
#     angle = np.abs(radians*180.0/np.pi)

#     if angle > 180.0:
#         angle =  360 - angle
#     return angle


labels_dict_left = {1: '001', 2: '002', 3: '003', 4: '004', 5:'005', 6:'006',7:'007',8:'008'}
labels_dict_right = {1: '001', 2: '002', 3: '003', 4: '004', 5:'005', 6:'006',7:'007',8:'008'}

data_left = []
massage_left = ''
data_right = []
massage_right = ''
data_ud_angle = []
massage_ud_angle = ''
data_bt_angle = []
massage_bt_angle = ''
# prev_hand_state = None
left_ypoint = ()
right_ypoint = ()
head_y = ()
chest_y = ()
mid_body_y = ()
hips_y = ()

while(cap.isOpened()):

    data_aux_left = []
    data_aux_right = []
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

            data_aux_left.append(x - min(x_left))
            data_aux_left.append(y - min(y_left))

        x1 = int(min(x_left) * W) - 10
        y1 = int(min(y_left) * H) - 10

        x2 = int(max(x_left) * W) - 10
        y2 = int(max(y_left) * H) - 10

        prediction = model_left.predict([np.asarray(data_aux_left)])
        predicted_character = labels_dict_left[int(prediction[0])]


        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)
        cv2.putText(frame, predicted_character, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2,
                    cv2.LINE_AA)
        
        data_left.append(predicted_character)
        massage_left = ''.join(data_left) 

    else :
        data_aux_left = [0.999] * 42  
        left_ypoint = 0.999

        prediction = model_left.predict([np.asarray(data_aux_left)])
        predicted_character = labels_dict_left[int(prediction[0])]       
        
        data_left.append(predicted_character)
        massage_left = ''.join(data_left) 

    ######### มือขวา #########
    if results_holistic.right_hand_landmarks is not None:
        right_hand_landmarks = results_holistic.right_hand_landmarks

        for i in range(len(right_hand_landmarks.landmark)):
            x = right_hand_landmarks.landmark[i].x
            y = right_hand_landmarks.landmark[i].y

            x_right.append(x)
            y_right.append(y)

            data_aux_right.append(x - min(x_right))
            data_aux_right.append(y - min(y_right))

        x1 = int(min(x_right) * W) - 10
        y1 = int(min(y_right) * H) - 10

        x2 = int(max(x_right) * W) - 10
        y2 = int(max(y_right) * H) - 10

        prediction = model_right.predict([np.asarray(data_aux_right)])
        predicted_character = labels_dict_right[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)
        cv2.putText(frame, predicted_character, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2,
                    cv2.LINE_AA)
        
        data_right.append(predicted_character)
        massage_right = ''.join(data_right)
    else:
        data_aux_right = [0.999] * 42 
        right_ypoint = 0.999

        prediction = model_right.predict([np.asarray(data_aux_right)])
        predicted_character = labels_dict_right[int(prediction[0])]

        data_right.append(predicted_character)
        massage_right = ''.join(data_right) 
        

         ########### แพ้อาหาร ########
    #มือซ้าย
    if '001002' and '002003' in massage_left :
        left_hand3 = True 
    else :
        left_hand3 = False
    # มือขวา
    if '001002' and '002003' in massage_right :
        right_hand3 = True 
    else :
        right_hand3 = False

    if left_hand3 == True and right_hand3 == True :
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.text((70, 5), "แพ้อาหาร", font=font, fill=(255, 255, 255))
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    
         ########### สบายดี ########
    #มือซ้าย
    if '004005' in massage_left :
        left_hand4 = True 
    else :
        left_hand4 = False
    # มือขวา
    if '004005' in massage_right :
        right_hand3 = True 
    else :
        right_hand4 = False

    if left_hand4 == True and right_hand4 == True :
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.text((70, 5), "สบายดี", font=font, fill=(255, 255, 255))
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
      break
    
cap.release()
cv2.destroyAllWindows()
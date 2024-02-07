import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image

font = ImageFont.truetype("./font/THSarabun.ttf", 50)

model_dict_left = pickle.load(open('./model_left.p', 'rb'))
model_left = model_dict_left['model']

model_dict_right = pickle.load(open('./model_right.p', 'rb'))
model_right = model_dict_right['model']


cap = cv2.VideoCapture()
cap.open('vdo/DDD.mov') 
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(
    static_image_mode=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

labels_dict = {1: '001', 2: '002', 3: '003'}
data__left = []
joined_data_left = ''
data_left = []
massage_left = ''
data__right = []
joined_data_right = ''
data_right = []
massage_right = ''
prev_hand_state = None

while(cap.isOpened()):

    data_aux = []
    x_left = []
    y_left = []
    x_right = []
    y_right = []

    ret, frame = cap.read()
    if ret == True :
     H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    # # หายตามมือ
    # hand_state = "OPEN" if results.multi_hand_landmarks else "CLOSED"
    # if hand_state != prev_hand_state:
    #     data__ = []
    #     joined_data = ''
    #     data_ = []
    #     massage = ''
    # prev_hand_state = hand_state


    ######### มือซ้าย #########
    if results.left_hand_landmarks is not None:
        left_hand_landmarks = results.left_hand_landmarks

        for i in range(len(left_hand_landmarks.landmark)):
            x = left_hand_landmarks.landmark[i].x
            y = left_hand_landmarks.landmark[i].y

            x_left.append(x)
            y_left.append(y)

        for i in range(len(left_hand_landmarks.landmark)):
            x = left_hand_landmarks.landmark[i].x
            y = left_hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_left))
            data_aux.append(y - min(y_left))

        x1 = int(min(x_left) * W) - 10
        y1 = int(min(y_left) * H) - 10

        x2 = int(max(x_left) * W) - 10
        y2 = int(max(y_left) * H) - 10

        prediction = model_left.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)
        cv2.putText(frame, predicted_character, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2,
                    cv2.LINE_AA)
        
        data_left.append(predicted_character)
        print(data_left)
        massage_left = ''.join(data_left) 
        print(massage_left)

        for i in range(0, len(predicted_character), 3):
            substring = predicted_character[i:i+3]
            if substring not in data__left:
                data__left.append(substring)

        joined_data_left = ''.join(data__left) 
        print(joined_data_left)

    #รหัส A คือหมวดท่าทาง 1
    A00 = '000'
    A01 = '001'
    A02 = '002'
    A03 = '003'
    A04 = '004'
    A05 = '005'
    A06 = '006'
    A07 = '007'
    A08 = '008'
    A10 = '010'
    A11 = '011'

    # รหัส Q คือคำแปล
    Q00 = '000'#ฉัน
    Q01 = '001'#คุณ
    Q02 = '002' + '003'#ตาย
    Q03 = '005' + '006'#อมยา
    Q04 = '006' + '008'#หมอ
    Q05 = '006' + '007'+'008'#หมอ
    Q051 = '006007007007007007007007008'
    Q06 = '009010009010' #เป็นหวัด
    Q07 = '011012' #ตาแดง
    Q08 = '011013' #มึนหัว
    Q09 = '001004' #ฉีดยา
    Q10 = '009010009010'
    print(joined_data_left)


    ######### มือขวา #########
    if results.right_hand_landmarks is not None:
        right_hand_landmarks = results.right_hand_landmarks

        for i in range(len(right_hand_landmarks.landmark)):
            x = right_hand_landmarks.landmark[i].x
            y = right_hand_landmarks.landmark[i].y

            x_right.append(x)
            y_right.append(y)

        for i in range(len(right_hand_landmarks.landmark)):
            x = right_hand_landmarks.landmark[i].x
            y = right_hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_right))
            data_aux.append(y - min(y_right))

        x1 = int(min(x_right) * W) - 10
        y1 = int(min(y_right) * H) - 10

        x2 = int(max(x_right) * W) - 10
        y2 = int(max(y_right) * H) - 10

        prediction = model_right.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)
        cv2.putText(frame, predicted_character, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2,
                    cv2.LINE_AA)
        
        data_right.append(predicted_character)
        print(data_right)
        massage_right = ''.join(data_right) 
        print(massage_right)

        for i in range(0, len(predicted_character), 3):
            substring = predicted_character[i:i+3]
            if substring not in data__right:
                data__right.append(substring)

        joined_data_right = ''.join(data__right) 
        print(joined_data_right)

        #รหัส A คือหมวดท่าทาง 1
    A00 = '000'
    A01 = '001'
    A02 = '002'
    A03 = '003'
    A04 = '004'
    A05 = '005'
    A06 = '006'
    A07 = '007'
    A08 = '008'
    A10 = '010'
    A11 = '011'

    # รหัส Q คือคำแปล
    Q000 = '000' + '001' + '002'
    Q00 = '000'#ฉัน
    Q01 = '001'#คุณ
    Q02 = '002' + '003'#ตาย
    Q03 = '005' + '006'#อมยา
    Q04 = '006' + '008'#หมอ
    Q05 = '006' + '007'+'008'#หมอ
    Q051 = '006007007007007007007007008'
    Q06 = '009010009010' #เป็นหวัด
    Q07 = '011012' #ตาแดง
    Q08 = '011013' #มึนหัว
    Q09 = '001004' #ฉีดยา
    Q10 = '009010009010'
    print(joined_data_right)





       
    ########## Diad ########
    if Q000 in joined_data_left :
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.text((70, 5), "ทดสอบ", font=font, fill=(255, 255, 255))
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        # print(f'Dead')

    # ######### อมยา ########
    # if Q03 in joined_data :
    #     frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #     draw = ImageDraw.Draw(frame_pil)
    #     draw.text((70, 5), "อมยา", font=font, fill=(255, 255, 255))
    #     frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    #     # print(f'take_medicine')

    # ######### หมอ ########
    # if Q04 in joined_data :
    #     frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #     draw = ImageDraw.Draw(frame_pil)
    #     draw.text((70, 5), "หมอ/แพทย์", font=font, fill=(255, 255, 255))
    #     frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    # ######### หมอ ########
    # if Q05 in joined_data :
    #     frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #     draw = ImageDraw.Draw(frame_pil)
    #     draw.text((70, 5), "หมอ/แพทย์", font=font, fill=(255, 255, 255))
    #     frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    # ######## หมอ ########
    # if Q051 in massage :
    #     frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #     draw = ImageDraw.Draw(frame_pil)
    #     draw.text((70, 5), "หมอ/แพทย์", font=font, fill=(255, 255, 255))
    #     frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    # ######### เป็นหวัด ########
    # if Q06 in massage :
    #     frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #     draw = ImageDraw.Draw(frame_pil)
    #     draw.text((70, 5), "เป็นหวัด/น้ำมูกไหล", font=font, fill=(255, 255, 255))
    #     frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # if Q10 in massage :
    #     frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #     draw = ImageDraw.Draw(frame_pil)
    #     draw.text((70, 5), "เป็นหวัด/น้ำมูกไหล", font=font, fill=(255, 255, 255))
    #     frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    # ######### ตาแดง ########
    # if Q07 in joined_data :
    #     frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #     draw = ImageDraw.Draw(frame_pil)
    #     draw.text((70, 5), "ตาแดง", font=font, fill=(255, 255, 255))
    #     frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    # ######### มึนหัว ########
    # # if Q08 in massage :
    # #     frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # #     draw = ImageDraw.Draw(frame_pil)
    # #     draw.text((70, 5), "มึนหัว/มึนศีรษะ", font=font, fill=(255, 255, 255))
    # #     frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    # ######### ฉีดยา ########
    # if Q09 in massage :
    #     frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #     draw = ImageDraw.Draw(frame_pil)
    #     draw.text((70, 5), "ฉีดยา", font=font, fill=(255, 255, 255))
    #     frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        
        

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
      break
    
cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import math
import requests

token = 'wG51kC97KFtqN0wKjJwXwC02XLvRcb3JdE7775C4jI3'

def send_msg(message, image_path):
    url = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': 'Bearer ' + token}
    data = {"message": message}
    with open(image_path, 'rb') as image_file:
        file = {'imageFile': image_file}
        requests.post(url, headers=headers, params=data, files=file)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def vector_2d_angle(v1, v2):
    try:
        angle = math.degrees(math.acos((v1[0]*v2[0] + v1[1]*v2[1]) / 
                                       ((v1[0]**2 + v1[1]**2)**0.5 * (v2[0]**2 + v2[1]**2)**0.5)))
    except:
        angle = 180
    return angle

def hand_angle(hand_):
    angles = []
    for i, (j1, j2) in enumerate([(0, 2), (0, 6), (0, 10), (0, 14), (0, 18)]):
        angle = vector_2d_angle(
            ((int(hand_[j1][0]) - int(hand_[j2][0])), (int(hand_[j1][1]) - int(hand_[j2][1]))),
            ((int(hand_[j2 + 1][0]) - int(hand_[j2 + 2][0])), (int(hand_[j2 + 1][1]) - int(hand_[j2 + 2][1])))
        )
        angles.append(angle)
    return angles

def hand_pos(finger_angle):
    f1, f2, f3, f4, f5 = finger_angle
    if f1 >= 50 and f2 < 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
        return '1'
    elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 >= 50 and f5 >= 50:
        return '2'
    elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 >= 50:
        return '3'
    else:
        return ''

cap = cv2.VideoCapture(0)
fontFace = cv2.FONT_HERSHEY_SIMPLEX
lineType = cv2.LINE_AA

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    w, h = 540, 310
    while True:
        ret, img = cap.read()
        img = cv2.resize(img, (w, h))
        if not ret:
            print("Cannot receive frame")
            break
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img2)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger_points = []
                for i in hand_landmarks.landmark:
                    x = int(i.x * w)
                    y = int(i.y * h)
                    finger_points.append((x, y))
                if finger_points:
                    angles = hand_angle(finger_points)
                    gesture = hand_pos(angles)
                    cv2.putText(img, gesture, (30, 120), fontFace, 2, (255, 255, 255), 3, lineType)

                    if gesture == '1':
                        cv2.imwrite('img/image.jpg', img)
                        send_msg('警告: 心臟病發!', 'img/image.jpg')
                    elif gesture == '2':
                        cv2.imwrite('img/image.jpg', img)
                        send_msg('注意: 氣喘發作!', 'img/image.jpg')
                    elif gesture == '3':
                        cv2.imwrite('img/image.jpg', img)
                        send_msg('警告: 頭痛!', 'img/image.jpg')

        cv2.imshow('view', img)
        if cv2.waitKey(5) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

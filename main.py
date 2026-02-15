import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# ================== CONFIG ==================
max_points = 1024
colors = [(255,0,0), (0,255,0), (0,0,255), (0,255,255)]
colorIndex = 0
is_paused = False

# Store strokes per color
points = [ [deque(maxlen=max_points)] for _ in range(4) ]
indices = [0,0,0,0]

# Smoothing variables
prev_x, prev_y = None, None
alpha = 0.7  # smoothing factor

# ================== MEDIAPIPE ==================
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    paintWindow = np.ones((480,640,3), dtype=np.uint8) * 255

    # ----------- UI BUTTONS -----------
    cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
    cv2.putText(paintWindow, "CLEAR", (49,33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    button_positions = [(160,255), (275,370), (390,485), (505,600)]
    for i in range(4):
        cv2.rectangle(paintWindow,
                      (button_positions[i][0],1),
                      (button_positions[i][1],65),
                      colors[i], -1)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            landmarks = []
            for lm in handLms.landmark:
                landmarks.append((int(lm.x*640), int(lm.y*480)))

            index_tip = landmarks[8]
            thumb_tip = landmarks[4]

            x, y = index_tip

            # Smooth movement
            if prev_x is not None:
                x = int(alpha * prev_x + (1-alpha) * x)
                y = int(alpha * prev_y + (1-alpha) * y)

            prev_x, prev_y = x, y
            center = (x,y)

            cv2.circle(frame, center, 6, (0,255,255), -1)

            distance = np.hypot(x - thumb_tip[0], y - thumb_tip[1])

            if not is_paused:

                # Pinch detected -> break stroke
                if distance < 30:
                    points[colorIndex].append(deque(maxlen=max_points))
                    indices[colorIndex] += 1
                    prev_x, prev_y = None, None

                # Button click zone
                elif y <= 65:
                    if 40 <= x <= 140:  # Clear
                        points = [ [deque(maxlen=max_points)] for _ in range(4) ]
                        indices = [0,0,0,0]

                    for i in range(4):
                        if button_positions[i][0] <= x <= button_positions[i][1]:
                            colorIndex = i

                # Draw normally
                else:
                    points[colorIndex][indices[colorIndex]].appendleft(center)

    else:
        prev_x, prev_y = None, None

    # ----------- DRAW STROKES -----------
    for i in range(4):
        for stroke in points[i]:
            for j in range(1, len(stroke)):
                if stroke[j-1] and stroke[j]:
                    cv2.line(paintWindow,
                             stroke[j-1],
                             stroke[j],
                             colors[i], 5)

    if is_paused:
        cv2.putText(paintWindow, "PAUSED",
                    (250,240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (100,100,100), 2)

    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('p'):
        is_paused = not is_paused
    elif key == ord('s'):
        cv2.imwrite("AirCanvas_Drawing.png", paintWindow)
        print("Image Saved!")

cap.release()
cv2.destroyAllWindows()

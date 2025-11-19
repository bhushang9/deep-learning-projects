import cv2
import mediapipe as mp
import csv
import os
import time

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

POSE_LABEL = "T_Pose"  #Change this to Hands_Up, Standing, etc.

cap = cv2.VideoCapture(0)
holistic = mp_holistic.Holistic()
output_file = open(f'{POSE_LABEL}.csv', mode='w', newline='')
csv_writer = csv.writer(output_file)

print("Collecting data for:", POSE_LABEL)
time.sleep(3)
print("Starting...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    if results.pose_landmarks:
        row = []
        for lm in results.pose_landmarks.landmark:
            row += [lm.x, lm.y, lm.z, lm.visibility]
        row.append(POSE_LABEL)
        csv_writer.writerow(row)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    cv2.imshow("Collecting", image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

output_file.close()
cap.release()
cv2.destroyAllWindows()

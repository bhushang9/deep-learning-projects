import cv2
import mediapipe as mp
import numpy as np
import pickle

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Load the trained model
with open("pose_model.pkl", "rb") as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)
holistic = mp_holistic.Holistic()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        row = []
        for lm in results.pose_landmarks.landmark:
            row += [lm.x, lm.y, lm.z, lm.visibility]

        X_input = np.array(row).reshape(1, -1)
        pose_class = model.predict(X_input)[0]

        # Display prediction
        cv2.putText(image, f'Pose: {pose_class}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Pose Classification", image)

    #with q the app will be quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

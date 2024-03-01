import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

# Load your trained model
model = load_model('asl_model.h5')

folder = "DATA/Z"
counter = 0

url = 'https://192.168.2.12:8080/video'
cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

desired_width = 300  # Adjust this based on your requirements
desired_height = 300  # Adjust this based on your requirements

while True:
    success, frame = cap.read()
    if frame is not None:
        frame_for_display = frame.copy()
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks and connections
                mp_drawing.draw_landmarks(
                    frame_for_display,
                    hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Calculate bounding box manually
                x_min = frame.shape[1]
                y_min = frame.shape[0]
                x_max = 0
                y_max = 0

                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                # Ensure the bounding box has a minimum size
                x_min, y_min = max(0, x_min - 10), max(0, y_min - 10)
                x_max, y_max = min(frame.shape[1], x_max + 10), min(frame.shape[0], y_max + 10)

                # Crop and resize the hand region
                if x_max > x_min and y_max > y_min:
                    img_crop = frame[y_min:y_max, x_min:x_max]
                    img_crop_resized = cv2.resize(img_crop, (desired_width, desired_height))

                    # Prepare the image for the model (normalize and reshape)
                    img_for_model = np.expand_dims(img_crop_resized, axis=0) / 255.0

                    # Predict the sign language letter
                    prediction = model.predict(img_for_model)
                    predicted_class = np.argmax(prediction, axis=1)
                    predicted_letter = chr(predicted_class[0] + 65)  # Assuming class 0 corresponds to 'A', 1 to 'B', etc.

                    # Draw bounding box and predicted letter
                    cv2.rectangle(frame_for_display, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame_for_display, predicted_letter, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Hand Tracking with Prediction', frame_for_display)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    # For image collection
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Z_{counter}.jpg', img_crop_resized)
        print(f"Saved {folder}/Z_{counter}.jpg")

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp


folder = "DATA/Z"
counter = 0

url = 'https://192.168.2.12:8080/video'
cap = cv2.VideoCapture(url)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

desired_width = 300  # Adjust this based on your requirements
desired_height = 300  # Adjust this based on your requirements

while True:
    success, frame = cap.read()
    if frame is not None:

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks and connections
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Calculate bounding box manually
                x_min= frame.shape[1]
                y_min= frame.shape[0] 
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

                # Crop the hand region
                img_crop = frame[y_min:y_max, x_min:x_max]

                # Resize the cropped image to the desired dimensions
                img_crop_resized = cv2.resize(img_crop, (desired_width, desired_height))

                cv2.imshow('Cropped Hand', img_crop_resized)

        cv2.imshow('Hand Tracking', frame)

    key = cv2.waitKey(1)
    
    if key == ord("q"):
        break
    #for image collection
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Z_{counter}.jpg', img_crop_resized)
        print(counter)


cap.release()
cv2.destroyAllWindows()
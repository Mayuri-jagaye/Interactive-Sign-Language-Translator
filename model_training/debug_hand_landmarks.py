import cv2
import mediapipe as mp
import sys
import os


def debug_hand_landmarks(image_path):
    # Check if file exists
    if not os.path.isfile(image_path):
        print(f"File does not exist: {image_path}")
        return

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read the image file: {image_path}")
        return

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run MediaPipe
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,  # or increase if multiple hands are expected
            min_detection_confidence=0.5
    ) as hands:
        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            print(f"No hand landmarks detected in: {image_path}")
            # return
        else:
            # Print the landmark coordinates
            print(f"Hand landmarks detected in: {image_path}")
            for hand_landmarks in results.multi_hand_landmarks:
                for i, lm in enumerate(hand_landmarks.landmark):
                    print(f"  Landmark {i}: (x={lm.x}, y={lm.y}, z={lm.z})")

                # Optional: Draw the landmarks on the image
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

    # If you want to see the drawn landmarks, uncomment:
    cv2.imshow('Debug Hand Landmarks', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_hand_landmarks.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    debug_hand_landmarks(image_path)

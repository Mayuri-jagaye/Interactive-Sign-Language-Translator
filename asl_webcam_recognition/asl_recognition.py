import cv2
import mediapipe as mp
import torch
import numpy as np
import time

# Load the trained model
model_path = "asl_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(model_path, map_location=device)
model_state = checkpoint["model_state_dict"]
label_to_idx = checkpoint["label_to_idx"]
idx_to_label = checkpoint["idx_to_label"]

num_classes = len(label_to_idx)

# Load the model architecture
class SimpleASLModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleASLModel(input_dim=63, hidden_dim=128, num_classes=num_classes).to(device)
model.load_state_dict(model_state)
model.eval()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Word capture variables
current_word = []
sign_start_time = None
current_sign = None
sign_hold_threshold = 2.0  # 2 seconds
last_captured_sign = None

# Button variables - positioned at bottom with smaller size
delete_button = {'x': 10, 'y': 400, 'w': 80, 'h': 30, 'text': 'DELETE'}
space_button = {'x': 100, 'y': 400, 'w': 80, 'h': 30, 'text': 'SPACE'}
clear_button = {'x': 190, 'y': 400, 'w': 80, 'h': 30, 'text': 'CLEAR'}

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global current_word
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if DELETE button is clicked
        if (delete_button['x'] <= x <= delete_button['x'] + delete_button['w'] and 
            delete_button['y'] <= y <= delete_button['y'] + delete_button['h']):
            if current_word:
                deleted_letter = current_word.pop()
                print(f"Deleted letter: {deleted_letter}")
            else:
                print("No letters to delete")
        
        # Check if SPACE button is clicked
        elif (space_button['x'] <= x <= space_button['x'] + space_button['w'] and 
              space_button['y'] <= y <= space_button['y'] + space_button['h']):
            current_word.append(" ")
            print("Added space")
        
        # Check if CLEAR button is clicked
        elif (clear_button['x'] <= x <= clear_button['x'] + clear_button['w'] and 
              clear_button['y'] <= y <= clear_button['y'] + clear_button['h']):
            current_word = []
            last_captured_sign = None
            print("Word cleared!")

# Open webcam
cap = cv2.VideoCapture(0)

# Set mouse callback
cv2.namedWindow("ASL Recognition")
cv2.setMouseCallback("ASL Recognition", mouse_callback)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = hands.process(rgb_frame)
    label_predicted = "Not sure"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks (flatten to match model input)
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            landmarks_np = np.array([coord for point in landmarks for coord in point], dtype=np.float32)

            # Convert to tensor
            landmarks_tensor = torch.tensor(landmarks_np).to(device).unsqueeze(0)  # Add batch dimension

            # Make prediction
            with torch.no_grad():
                output = model(landmarks_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, dim=1)

            confidence = confidence.item()
            predicted_label = idx_to_label[predicted_idx.item()]

            # Confidence threshold
            if confidence > 0.6:
                label_predicted = predicted_label
                
                # Handle sign timing for word capture (only for letters, not special functions)
                if predicted_label != current_sign:
                    # New sign detected
                    current_sign = predicted_label
                    sign_start_time = time.time()
                elif sign_start_time and (time.time() - sign_start_time) >= sign_hold_threshold:
                    # Sign held for 2+ seconds, capture it (only letters A-Z)
                    if predicted_label.isalpha():  # Only capture alphabetic characters
                        current_word.append(predicted_label)
                        print(f"Captured letter: {predicted_label}")
                        last_captured_sign = predicted_label
                        # Reset timing to allow consecutive captures
                        sign_start_time = time.time()
            else:
                # No confident prediction, reset timing
                current_sign = None
                sign_start_time = None
    else:
        # No hand detected, reset timing
        current_sign = None
        sign_start_time = None

    # Calculate remaining time for current sign
    time_remaining = 0
    if sign_start_time and current_sign:
        elapsed = time.time() - sign_start_time
        if elapsed < sign_hold_threshold:
            time_remaining = sign_hold_threshold - elapsed

    # Display predicted label
    cv2.putText(frame, f"Prediction: {label_predicted}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Display timer for current sign
    if time_remaining > 0:
        timer_text = f"Hold for: {time_remaining:.1f}s"
        cv2.putText(frame, timer_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Display current word
    word_text = f"Word: {''.join(current_word)}"
    cv2.putText(frame, word_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw buttons
    # DELETE button (red)
    cv2.rectangle(frame, (delete_button['x'], delete_button['y']), 
                  (delete_button['x'] + delete_button['w'], delete_button['y'] + delete_button['h']), 
                  (0, 0, 255), -1)
    cv2.putText(frame, delete_button['text'], 
                (delete_button['x'] + 5, delete_button['y'] + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # SPACE button (blue)
    cv2.rectangle(frame, (space_button['x'], space_button['y']), 
                  (space_button['x'] + space_button['w'], space_button['y'] + space_button['h']), 
                  (255, 0, 0), -1)
    cv2.putText(frame, space_button['text'], 
                (space_button['x'] + 15, space_button['y'] + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # CLEAR button (orange)
    cv2.rectangle(frame, (clear_button['x'], clear_button['y']), 
                  (clear_button['x'] + clear_button['w'], clear_button['y'] + clear_button['h']), 
                  (0, 165, 255), -1)
    cv2.putText(frame, clear_button['text'], 
                (clear_button['x'] + 15, clear_button['y'] + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("ASL Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

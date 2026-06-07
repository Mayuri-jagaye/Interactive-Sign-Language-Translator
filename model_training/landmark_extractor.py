# landmark_extractor.py

import cv2
import mediapipe as mp
from pathlib import Path

def extract_landmarks_from_image(image_path, hands):
    """
    Extracts hand landmarks from an image using MediaPipe Hands.
    Returns a list of (x, y, z) tuples or None if no hand is detected.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error reading image: {image_path}")
        return None
    # Convert BGR image to RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        # Assuming one hand per image.
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        return landmarks
    else:
        return None

def process_dataset(root_dir):
    """
    Walks through the dataset folder, extracts landmarks from each image,
    and returns a list of dictionaries containing file path, label, split, and landmarks.
    Prints progress messages to help debug issues with detection.
    """
    root = Path(root_dir)
    landmarks_data = []
    mp_hands = mp.solutions.hands

    total_processed = 0
    total_detected = 0

    # Configure MediaPipe for static image processing.
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        for split in ['asl_alphabet_test', 'asl_alphabet_train']:
            split_dir = root / split
            if not split_dir.exists():
                print(f"Skipping non-existing directory: {split_dir}")
                continue

            # For the test set: images are directly in the split folder.
            if split == 'asl_alphabet_test':
                image_files = list(split_dir.glob('*.jpg'))
                for image_file in image_files:
                    total_processed += 1
                    label = image_file.stem.split('_')[0]
                    landmarks = extract_landmarks_from_image(image_file, hands)

                    if landmarks is not None:
                        total_detected += 1
                        print(f"[{total_processed}] Landmarks found in '{image_file.name}' (label: {label})")
                    else:
                        print(f"[{total_processed}] No landmarks found in '{image_file.name}' (label: {label})")

                    landmarks_data.append({
                        'file_path': str(image_file),
                        'split': split,
                        'label': label,
                        'landmarks': landmarks
                    })
            else:
                # For the training set: images are inside subfolders named by their label.
                label_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
                for label_dir in label_dirs:
                    label = label_dir.name
                    image_files = list(label_dir.glob('*.jpg'))
                    for image_file in image_files:
                        total_processed += 1
                        landmarks = extract_landmarks_from_image(image_file, hands)

                        if landmarks is not None:
                            total_detected += 1
                            print(f"[{total_processed}] Landmarks found in '{image_file.name}' (label: {label})")
                        else:
                            print(f"[{total_processed}] No landmarks found in '{image_file.name}' (label: {label})")

                        landmarks_data.append({
                            'file_path': str(image_file),
                            'split': split,
                            'label': label,
                            'landmarks': landmarks
                        })

    print("\n=== Summary ===")
    print(f"Total images processed: {total_processed}")
    print(f"Total images with detected landmarks: {total_detected}")
    print(f"Total images with no landmarks: {total_processed - total_detected}")

    return landmarks_data

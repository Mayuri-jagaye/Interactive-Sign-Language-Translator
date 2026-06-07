# main.py

from landmark_extractor import process_dataset
from landmarks_db import store_landmarks_in_db


def main():
    dataset_root = '../asl_alphabet_image_dataset'
    print("Processing dataset to extract landmarks...")
    landmarks_data = process_dataset(dataset_root)
    print(f"Extracted landmarks from {len(landmarks_data)} images.")

    print("Storing landmarks into SQLite database...")
    store_landmarks_in_db(landmarks_data)
    print("Landmarks stored in landmarks.db")


if __name__ == '__main__':
    main()

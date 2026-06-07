# American Sign Language (ASL) Recognition System

A multi-component machine learning application for real-time American Sign Language (ASL) alphabet recognition. The system leverages **MediaPipe Hands** for robust hand tracking and joint landmark extraction, **PyTorch** for deep learning-based gesture classification, and **Flask / OpenCV** for interactive real-time webcam interfaces.

---

## 📂 Project Structure

```text
.
├── asl_webcam_recognition/     # Standalone OpenCV desktop webcam application
│   ├── asl_recognition.py      # Real-time desktop prediction & interactive UI
│   └── hand_tracking.py        # MediaPipe hand tracking extraction demo
│
├── backend_web/                # Flask web backend for video stream & state API
│   ├── web_server.py           # Flask server with video feed and REST endpoints
│   └── requirements.txt        # Web server dependencies
│
└── model_training/             # Data preprocessing, database storage, & training pipeline
    ├── landmark_extractor.py   # Extracts 21 landmarks (x,y,z) from dataset images
    ├── landmarks_db.py         # Stores extracted landmarks into an SQLite database
    ├── landmarks.py            # Main entry point to process images and store in DB
    ├── db_to_df.py             # Prepares Pandas DataFrames and splits training data
    ├── train_model.py          # PyTorch feed-forward neural network training script
    └── debug_hand_landmarks.py # Utility script to debug landmarks on a single image
```

---

## ⚙️ How It Works

1. **Landmark Extraction**: Hand postures are captured via MediaPipe Hands, which detects **21 key joints** (x, y, z coordinates) on a single hand.
2. **Model Training**: A 63-dimensional coordinate vector (21 landmarks * 3 coordinates) is fed into a 3-layer Feed-Forward PyTorch Neural Network (`SimpleASLModel`), classifying the hand pose into the corresponding ASL alphabet letter (`A-Z`) or `nothing`.
3. **Real-time Recognition**: The trained model classifies gestures in real-time from your webcam. By holding a gesture for a configurable hold threshold (default: **2.0 seconds**), the character is locked and appended to the current word.
4. **Interactive Controls**: Users can delete characters, add spaces, or clear the word using on-screen overlay buttons (OpenCV UI) or REST API requests (Web app).

---

## 🛠️ Setup & Usage Instructions

### 1. Model Training & Data Preparation (`model_training/`)
This module is responsible for extracting hand landmarks from an image dataset, storing them in an SQLite database, and training the classification model.

* **Dataset Setup**: The system expects the ASL alphabet image dataset to be structured in a folder named `asl_alphabet_image_dataset` in the parent directory:
  ```text
  asl_alphabet_image_dataset/
  ├── asl_alphabet_train/
  │   ├── A/
  │   ├── B/
  │   ...
  └── asl_alphabet_test/
  ```

* **Step 1: Landmark Extraction**: Run `landmarks.py` to extract hand coordinates from dataset images and store them in `landmarks.db`.
  ```bash
  cd model_training
  python landmarks.py
  ```

* **Step 2: Train PyTorch Model**: Run `train_model.py` to load data from `landmarks.db`, perform a train/test split (moving 5% of training data to the test set), train the PyTorch MLP model, and output the model weights as `asl_model.pth` in the parent directory.
  ```bash
  python train_model.py
  ```

* **Debugging Landmark Detection**: You can visualize hand landmark extraction on a single image file by running:
  ```bash
  python debug_hand_landmarks.py <path_to_image.jpg>
  ```

---

### 2. Standalone Webcam Application (`asl_webcam_recognition/`)
This is a native Python desktop application using OpenCV and MediaPipe.

* **Prerequisites**: Make sure the trained `asl_model.pth` file is located in this directory or the root directory.
* **Execution**: Launch the GUI webcam window:
  ```bash
  cd asl_webcam_recognition
  python asl_recognition.py
  ```
* **Interactive UI Controls**:
  * **On-screen buttons** (clickable via mouse directly on the video window):
    * `DELETE` (Red): Deletes the last character from the current word.
    * `SPACE` (Blue): Appends a space character.
    * `CLEAR` (Orange): Resets the current word.
  * **Keyboard controls**:
    * Press `q` to quit the application.

---

### 3. Flask Web Server (`backend_web/`)
The web backend streams the webcam feed and provides endpoints for a web-based frontend application.

* **Installation**: Install the required Flask packages:
  ```bash
  cd backend_web
  pip install -r requirements.txt
  ```
* **Execution**: Place the trained `asl_model.pth` in this directory and start the server:
  ```bash
  python web_server.py
  ```
  The server will start running locally at `http://localhost:5000`.

* **REST API & Endpoints**:
  * `GET /health` - Checks if the server is running.
  * `GET /video` - Serves a live MJPEG video stream from the webcam with overlaid predictions and hand tracking lines.
  * `GET /state` - Retrieves current predictions, confidence scores, accumulated word, and remaining timer values in JSON format.
  * `POST /action` - Accepts JSON payloads to trigger actions:
    * `{"type": "delete"}` - Delete last letter.
    * `{"type": "space"}` - Insert a space.
    * `{"type": "clear"}` - Clear current word.

---

## 🧠 Technical Specifications
* **MediaPipe Hands**: Used in real-time tracking mode (`static_image_mode=False`) for the webcam interface and static mode (`static_image_mode=True`) for the dataset landmark extractor.
* **Classifier Model Architecture**:
  * **Input Layer**: 63 features (21 landmarks × 3 coordinates `x`, `y`, `z`)
  * **Hidden Layers**: 2 fully-connected layers (128 hidden nodes each) with ReLU activation
  * **Output Layer**: Fully-connected layer outputting raw logits for each class
* **Classification Pipeline**:
  `Webcam Frame / Image -> MediaPipe Landmark Extraction -> Tensor Conversion -> PyTorch Model Prediction -> Softmax Probability Calculation -> Index Mapping to ASL Letter label`

import threading
import time
from typing import Generator, Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
from flask import Flask, Response, jsonify, request
from flask_cors import CORS


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint["model_state_dict"]
    idx_to_label = checkpoint["idx_to_label"]

    class SimpleASLModel(torch.nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = torch.nn.Linear(hidden_dim, num_classes)
            self.relu = torch.nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    num_classes = len(idx_to_label)
    model = SimpleASLModel(input_dim=63, hidden_dim=128, num_classes=num_classes).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return model, idx_to_label


class RecognitionState:
    def __init__(self) -> None:
        self.current_word = []  # list[str]
        self.current_sign: Optional[str] = None
        self.sign_start_time: Optional[float] = None
        self.sign_hold_threshold = 2.0
        self.last_prediction: str = "Not sure"
        self.last_confidence: float = 0.0
        self.time_remaining: float = 0.0
        self._lock = threading.Lock()

    def reset_timing(self) -> None:
        self.current_sign = None
        self.sign_start_time = None
        self.time_remaining = 0.0

    def to_dict(self):
        with self._lock:
            return {
                "prediction": self.last_prediction,
                "confidence": self.last_confidence,
                "currentWord": "".join(self.current_word),
                "timeRemaining": self.time_remaining,
            }

    def action_delete(self) -> None:
        with self._lock:
            if self.current_word:
                self.current_word.pop()

    def action_space(self) -> None:
        with self._lock:
            self.current_word.append(" ")

    def action_clear(self) -> None:
        with self._lock:
            self.current_word = []
            self.current_sign = None
            self.sign_start_time = None
            self.time_remaining = 0.0


class ASLRecognizer(threading.Thread):
    def __init__(self, checkpoint_path: str, state: RecognitionState, camera_index: int = 0):
        super().__init__(daemon=True)
        self.state = state
        self.camera_index = camera_index
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.idx_to_label = load_model(checkpoint_path, self.device)

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
        )

        self.cap = cv2.VideoCapture(self.camera_index)
        self._running = True
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

    def run(self) -> None:
        while self._running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.hands.process(rgb_frame)
            label_predicted = "Not sure"
            confidence_value = 0.0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )

                    landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    landmarks_np = np.array(
                        [coord for point in landmarks for coord in point], dtype=np.float32
                    )

                    landmarks_tensor = (
                        torch.tensor(landmarks_np).to(self.device).unsqueeze(0)
                    )

                    with torch.no_grad():
                        output = self.model(landmarks_tensor)
                        probabilities = torch.softmax(output, dim=1)
                        confidence, predicted_idx = torch.max(probabilities, dim=1)

                    confidence_value = float(confidence.item())
                    predicted_label = self.idx_to_label[int(predicted_idx.item())]

                    if confidence_value > 0.6:
                        label_predicted = predicted_label
                        with self.state._lock:
                            if predicted_label != self.state.current_sign:
                                self.state.current_sign = predicted_label
                                self.state.sign_start_time = time.time()
                            elif (
                                self.state.sign_start_time
                                and (time.time() - self.state.sign_start_time)
                                >= self.state.sign_hold_threshold
                            ):
                                if predicted_label.isalpha():
                                    self.state.current_word.append(predicted_label)
                                    self.state.sign_start_time = time.time()
                    else:
                        self.state.reset_timing()
            else:
                self.state.reset_timing()

            with self.state._lock:
                self.state.last_prediction = label_predicted
                self.state.last_confidence = confidence_value
                if self.state.sign_start_time and self.state.current_sign:
                    elapsed = time.time() - self.state.sign_start_time
                    if elapsed < self.state.sign_hold_threshold:
                        self.state.time_remaining = self.state.sign_hold_threshold - elapsed
                    else:
                        self.state.time_remaining = 0.0
                else:
                    self.state.time_remaining = 0.0

            # Overlay minimal info on frame for the stream
            cv2.putText(
                frame,
                f"Prediction: {label_predicted}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )
            with self.state._lock:
                word_text = f"Word: {''.join(self.state.current_word)}"
            cv2.putText(
                frame,
                word_text,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
            if self.state.time_remaining > 0:
                cv2.putText(
                    frame,
                    f"Hold for: {self.state.time_remaining:.1f}s",
                    (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            with self._frame_lock:
                self._latest_frame = frame

        if self.cap:
            self.cap.release()

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            ret, jpeg = cv2.imencode(".jpg", self._latest_frame)
            if not ret:
                return None
            return jpeg.tobytes()

    def stop(self) -> None:
        self._running = False


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    state = RecognitionState()
    recognizer = ASLRecognizer(checkpoint_path="asl_model.pth", state=state, camera_index=0)
    recognizer.start()

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/state")
    def get_state():
        return jsonify(state.to_dict())

    @app.post("/action")
    def post_action():
        payload = request.get_json(silent=True) or {}
        action_type = str(payload.get("type", "")).lower()
        if action_type == "delete":
            state.action_delete()
        elif action_type == "space":
            state.action_space()
        elif action_type == "clear":
            state.action_clear()
        else:
            return jsonify({"ok": False, "error": "unknown action"}), 400
        return jsonify({"ok": True, "currentWord": "".join(state.current_word)})

    @app.route("/video")
    def video_feed():
        def frame_generator() -> Generator[bytes, None, None]:
            boundary = b"--frame"
            while True:
                jpeg = recognizer.get_latest_jpeg()
                if jpeg is None:
                    time.sleep(0.01)
                    continue
                yield (
                    boundary
                    + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )

        return Response(
            frame_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return app


if __name__ == "__main__":
    import os
    app = create_app()
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)



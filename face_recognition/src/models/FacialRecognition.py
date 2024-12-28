import os
import cv2
import time
import pickle
import numpy as np
import face_recognition
from helpers.config import get_settings

class FacialRecognition:
    def __init__(self, encodings_path, video_source):
        self.encodings_path = encodings_path
        self.video_source = video_source
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.cv_scaler = 4
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.video_capture = None

    def load_encodings(self):
        """Load pre-trained face encodings from a pickle file."""
        print("[INFO] Loading encodings...")
        with open(self.encodings_path, "rb") as f:
            data = pickle.loads(f.read())
            self.known_face_encodings = data["encodings"]
            self.known_face_names = data["names"]
        print(f"[INFO] Loaded {len(self.known_face_encodings)} known faces.")

    def initialize_camera(self):
        """Initialize the video capture."""
        self.video_capture = cv2.VideoCapture(self.video_source)
        if not self.video_capture.isOpened():
            raise Exception("Error: Unable to open video source.")

    def process_frame(self, frame):
        """Process a single frame to detect and recognize faces."""
        resized_frame = cv2.resize(frame, (0, 0), fx=(1 / self.cv_scaler), fy=(1 / self.cv_scaler))
        rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        self.face_locations = face_recognition.face_locations(rgb_resized_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_resized_frame, self.face_locations, model='large')

        self.face_names = []
        for face_encoding in self.face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            if matches:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            self.face_names.append(name)

        return frame

    def draw_results(self, frame):
        """Draw results (face boxes and labels) on the frame."""
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            top *= self.cv_scaler
            right *= self.cv_scaler
            bottom *= self.cv_scaler
            left *= self.cv_scaler

            cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
            cv2.rectangle(frame, (left - 3, top - 35), (right + 3, top), (244, 42, 3), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)

        return frame

    def calculate_fps(self):
        """Calculate the current FPS."""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        return self.fps

    def run(self):
        """Run the facial recognition system."""
        try:
            self.load_encodings()
            self.initialize_camera()
            print("[INFO] Starting video stream...")

            while True:
                ret, frame = self.video_capture.read()
                if not ret:
                    print("Error: Unable to capture video.")
                    break

                processed_frame = self.process_frame(frame)
                display_frame = self.draw_results(processed_frame)

                current_fps = self.calculate_fps()
                cv2.putText(display_frame, f"FPS: {current_fps:.1f}",
                            (display_frame.shape[1] - 150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow('Video', display_frame)

                if cv2.waitKey(1) == ord("q"):
                    break

        finally:
            self.video_capture.release()
            cv2.destroyAllWindows()
            print("[INFO] Video stream stopped.")


if __name__ == "__main__":
    # Initialize and run the FacialRecognition system
    settings = get_settings()
    facial_recognition = FacialRecognition(
        encodings_path="face_recognition/src/assets/encodings.pickle",
        video_source=settings.CAMER_INPUT
    )
    facial_recognition.run()

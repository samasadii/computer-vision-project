import cv2
from mtcnn import MTCNN
import numpy as np


# The FaceDetector class provides methods for detection, tracking, and alignment of faces.
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    def __init__(self, tm_window_size=20, tm_threshold=0.7, aligned_image_size=224):
        # Prepare face alignment.
        self.detector = MTCNN()

        # Reference (initial face detection) for template matching.
        self.reference = None

        # Size of face image after landmark-based alignment.
        self.aligned_image_size = aligned_image_size
        self.reference_point = None

	# ToDo: Specify all parameters for template matching.
        self.tm_threshold = tm_threshold
        self.tm_window_size = tm_window_size

    # ToDo: Track a face in a new image using template matching.
    def track_face(self, image):
        if self.reference is None:
            return None  # No face was detected initially.

        # Define the search window around the last known position
        x, y = self.reference_point
        left = max(0, x - self.tm_window_size)
        right = min(image.shape[1], x + self.tm_window_size)
        top = max(0, y - self.tm_window_size)
        bottom = min(image.shape[0], y + self.tm_window_size)
        search_window = image[top:bottom, left:right]

        # Template matching
        result = cv2.matchTemplate(search_window, self.reference, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val < self.tm_threshold:
            return self.detect_face(image)  # Reinitialize tracking

        # Update the reference point
        self.reference_point = (left + max_loc[0] + self.reference.shape[1] // 2,
                                top + max_loc[1] + self.reference.shape[0] // 2)

        face_rect = [left + max_loc[0], top + max_loc[1], self.reference.shape[1], self.reference.shape[0]]
        aligned = self.align_face(image, face_rect)
        return {"rect": face_rect, "image": image, "aligned": aligned, "response": max_val}

    # Face detection in a new image.
    def detect_face(self, image):
        # Retrieve all detectable faces in the given image.
        detections = self.detector.detect_faces(image)
        if not detections:
            self.reference = None
            return None

        # Select face with largest bounding box.
        largest_detection = np.argmax([d["box"][2] * d["box"][3] for d in detections])
        face_rect = detections[largest_detection]["box"]

        # Align the detected face.
        aligned = self.align_face(image, face_rect)
        return {"rect": face_rect, "image": image, "aligned": aligned, "response": 0}

    # Face alignment to predefined size.
    def align_face(self, image, face_rect):
        return cv2.resize(self.crop_face(image, face_rect), dsize=(self.aligned_image_size, self.aligned_image_size))

    # Crop face according to detected bounding box.
    def crop_face(self, image, face_rect):
        top = max(face_rect[1], 0)
        left = max(face_rect[0], 0)
        bottom = min(face_rect[1] + face_rect[3] - 1, image.shape[0] - 1)
        right = min(face_rect[0] + face_rect[2] - 1, image.shape[1] - 1)
        return image[top:bottom, left:right, :]


import cv2
from mtcnn import MTCNN
import numpy as np


# The FaceDetector class provides methods for detection, tracking, and alignment of faces.
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    def __init__(self, tm_window_size=25, tm_threshold=0.65, aligned_image_size=224):
        # Prepare face alignment.
        self.detector = MTCNN()

        # Reference (initial face detection) for template matching.
        self.reference = None

        # Size of face image after landmark-based alignment.
        self.aligned_image_size = aligned_image_size

	# ToDo: Specify all parameters for template matching.
        self.tm_threshold = tm_threshold
        self.tm_window_size = tm_window_size

    # ToDo: Track a face in a new image using template matching.
    def track_face(self, image):
        # Check if there is a previously detected face to track
        if self.reference is None:
            detected_face = self.detect_face(image)
            if detected_face:
                self.reference = detected_face
            return detected_face

        # Compute the search region boundaries around the last known position of the detected face
        bounding_box = self.reference['rect']
        region_left = max(bounding_box[0] - self.tm_window_size, 0)
        region_top = max(bounding_box[1] - self.tm_window_size, 0)
        region_right = min(bounding_box[0] + bounding_box[2] + self.tm_window_size, image.shape[1])
        region_bottom = min(bounding_box[1] + bounding_box[3] + self.tm_window_size, image.shape[0])
        region_of_interest = image[region_top:region_bottom, region_left:region_right]

        grey_region_of_interest = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2GRAY)
        grey_aligned_face = cv2.cvtColor(self.crop_face(self.reference['image'], self.reference['rect']), cv2.COLOR_BGR2GRAY)

        # Use template matching to find the face in the new image section
        match_result = cv2.matchTemplate(grey_region_of_interest, grey_aligned_face, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)

        # Update the face reference if the found match is above the threshold
        if max_val >= self.tm_threshold:
            updated_rect = (
                max_loc[0] + region_left,
                max_loc[1] + region_top,
                self.reference['rect'][2],
                self.reference['rect'][3]
            )
            updated_aligned_face = self.align_face(image, updated_rect)
            self.reference = {"rect": updated_rect, "image": image, "aligned": updated_aligned_face, "response": max_val}
            return self.reference

        # If the correlation is below the threshold, attempt to redetect the face
        self.reference = self.detect_face(image)
        return self.reference

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


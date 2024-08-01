import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
from skimage.io import imread
from skimage.draw import rectangle_perimeter
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog
import joblib
from selective_search import selective_search

def run_inference(image_folder, annotations_file, model_path):
    # Define the minimum dimension for resizing
    min_dim = 64

    # Load the trained SVM model and the PCA model
    model = joblib.load(model_path)
    pca = joblib.load("results/pca_model.pkl")

    # Load the annotations file for the test data (if required)
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_folder, filename)
            image = imread(image_path)
            print(f"Processing image: {image_path}")

            # Generate region proposals
            _, regions = selective_search(image, scale=500, sigma=0.9, min_size=10)

            for region in regions:
                box = [region['rect'][0], region['rect'][1], region['rect'][2], region['rect'][3]]
                crop = image[box[1]:box[3], box[0]:box[2]]

                # Ensure the cropped image has valid dimensions
                if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                    print(f"Invalid crop dimensions for image {filename} with box {box}")
                    continue

                # Convert to grayscale if the image has multiple channels
                if crop.ndim == 3:
                    crop = rgb2gray(crop)

                # Check for valid dimensions after conversion and resize if necessary
                if crop.shape[0] < min_dim or crop.shape[1] < min_dim:
                    print(f"Resizing cropped image for {filename} with box {box}")
                    try:
                        crop_resized = resize(crop, (min_dim, min_dim), mode='reflect', anti_aliasing=True)
                    except ValueError as e:
                        print(f"Skipping region due to resize error: {e}")
                        continue
                else:
                    crop_resized = crop

                # Extract HOG features
                features = hog(crop_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
                features = features.flatten()  # Flatten the features

                # Check if the features match the expected size
                expected_feature_size = 1764  # Set this based on the training configuration
                if features.shape[0] != expected_feature_size:
                    print(f"Skipping region with unexpected feature size: {features.shape[0]}")
                    continue

                # Apply PCA transformation
                features_pca = pca.transform([features])[0]

                # Predict using the trained model
                prediction = model.predict([features_pca])[0]
                if prediction == 1:  # If the prediction is 'balloon'
                    rr, cc = rectangle_perimeter((box[1], box[0]), end=(box[3], box[2]), shape=image.shape)
                    image[rr, cc] = 255  # Mark the detected region

            # Display the image with detected regions
            plt.imshow(image)
            plt.show()

if __name__ == "__main__":
    run_inference("data/test", "data/test/_annotations.coco.json", "results/svm_model.pkl")

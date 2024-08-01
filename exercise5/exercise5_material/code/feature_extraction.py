import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.io import imread
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize

def extract_features(samples_path):
    with open(samples_path, 'rb') as f:
        samples = pickle.load(f)
    
    features = []
    labels = []
    min_dim = 64  # Ensuring all images are resized to a standard size

    for (filename, box, label) in samples:
        if os.path.exists(os.path.join("data", "train", filename)):
            image_path = os.path.join("data", "train", filename)
        elif os.path.exists(os.path.join("data", "valid", filename)):
            image_path = os.path.join("data", "valid", filename)
        else:
            print(f"Error: Image {filename} not found in train or valid folders.")
            continue

        print(f"Processing image: {image_path}")

        image = imread(image_path)

        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            print(f"Invalid box dimensions for image {filename}: {box}")
            continue

        cropped_image = image[y1:y2, x1:x2]
        if cropped_image.size == 0:
            print(f"Zero-size cropped image for {filename}: {box}")
            continue

        if cropped_image.ndim == 3:
            cropped_image = rgb2gray(cropped_image)

        crop_resized = resize(cropped_image, (min_dim, min_dim), mode='reflect', anti_aliasing=True)
        hog_features = hog(crop_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        features.append(hog_features.flatten())
        labels.append(1 if label == 'balloon' else 0)
    
    features = np.array(features)
    labels = np.array(labels)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=100)
    features_pca = pca.fit_transform(features_scaled)

    features_path = "results/features_labels.pkl"
    with open(features_path, 'wb') as f:
        pickle.dump((features_pca, labels, pca), f)  # Save PCA along with features and labels

    return features_path

if __name__ == "__main__":
    extract_features("results/training_samples.pkl")

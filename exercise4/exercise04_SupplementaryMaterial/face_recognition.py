import cv2
import numpy as np
import pickle
import os
import random
from scipy import spatial


# FaceNet to extract face embeddings.
class FaceNet:

    def __init__(self):
        self.dim_embeddings = 128
        self.facenet = cv2.dnn.readNetFromONNX("resnet50_128.onnx")

    # Predict embedding from a given face image.
    def predict(self, face):
        # Normalize face image using mean subtraction.
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) - (131.0912, 103.8827, 91.4953)

        # Forward pass through deep neural network. The input size should be 224 x 224.
        reshaped = np.moveaxis(face, 2, 0)
        reshaped = np.expand_dims(reshaped, axis=0)
        self.facenet.setInput(reshaped)
        embedding = np.squeeze(self.facenet.forward())
        return embedding / np.linalg.norm(embedding)

    # Get dimensionality of the extracted embeddings.
    def get_embedding_dimensionality(self):
        return self.dim_embeddings


# The FaceRecognizer model enables supervised face identification.
class FaceRecognizer:

    # Prepare FaceRecognizer; specify all parameters for face identification.
    def __init__(self, num_neighbours=11, max_distance=0.8, min_prob=0.5):
        # ToDo: Prepare FaceNet and set all parameters for kNN.
        self.facenet = FaceNet()
        self.num_neighbours = num_neighbours
        self.max_distance = max_distance
        self.min_prob = min_prob

        # The underlying gallery: class labels and embeddings.
        self.labels = []
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))

        # Load face recognizer from pickle file if available.
        if os.path.exists("recognition_gallery.pkl"):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open("recognition_gallery.pkl", 'wb') as f:
            pickle.dump((self.labels, self.embeddings), f)

    # Load trained model from a pickle file.
    def load(self):
        with open("recognition_gallery.pkl", 'rb') as f:
            (self.labels, self.embeddings) = pickle.load(f)

    # ToDo
    def update(self, face, label):
        embedding = self.facenet.predict(face)
        self.embeddings = np.vstack([self.embeddings, embedding])
        self.labels.append(label)

    # ToDo
    def predict(self, face):
        # Extract facial features and calculate distances to known embeddings
        feature_vector = self.facenet.predict(face)
        euclidean_distances = spatial.distance.cdist([feature_vector], self.embeddings, 'euclidean').flatten()

        # Identify the closest matches based on the computed distances
        sorted_indices = np.argsort(euclidean_distances)[:self.num_neighbours]
        labels_of_nearest = [self.labels[idx] for idx in sorted_indices]

        # Determine the most frequent label from the nearest neighbors
        label_frequencies = {label: labels_of_nearest.count(label) for label in set(labels_of_nearest)}
        predominant_label, highest_frequency = max(label_frequencies.items(), key=lambda item: item[1])
        label_probability = highest_frequency / self.num_neighbours
        
        # Calculate the minimum distance for the predominant label
        min_distance_for_label = min(euclidean_distances[i] for i, label in enumerate(self.labels) if label == predominant_label and i in sorted_indices)

        if label_probability < self.min_prob or min_distance_for_label > self.max_distance:
            return "unknown", label_probability, min_distance_for_label

        return predominant_label, label_probability, min_distance_for_label


# The FaceClustering class enables unsupervised clustering of face images according to their identity and
# re-identification.
class FaceClustering:

    # Prepare FaceClustering; specify all parameters of clustering algorithm.
    def __init__(self,num_clusters=2, max_iter=25):
        # ToDo: Prepare FaceNet.
        self.facenet = FaceNet()

        # The underlying gallery: embeddings without class labels.
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))

        # Number of cluster centers for k-means clustering.
        self.num_clusters = num_clusters
        # Cluster centers.
        self.cluster_center = np.empty((num_clusters, self.facenet.get_embedding_dimensionality()))
        # Cluster index associated with the different samples.
        self.cluster_membership = []

        # Maximum number of iterations for k-means clustering.
        self.max_iter = max_iter

        # Load face clustering from pickle file if available.
        if os.path.exists("clustering_gallery.pkl"):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open("clustering_gallery.pkl", 'wb') as f:
            pickle.dump((self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership), f)

    # Load trained model from a pickle file.
    def load(self):
        with open("clustering_gallery.pkl", 'rb') as f:
            (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = pickle.load(f)

    # ToDo
    def update(self, face):
        embedding = self.facenet.predict(face)
        self.embeddings = np.vstack([self.embeddings, embedding])
    
    # ToDo
    def fit(self):
        # Initialize cluster centers randomly from the existing embeddings
        initial_selection = np.random.permutation(self.embeddings.shape[0])[:self.num_clusters]
        self.cluster_centers = self.embeddings[initial_selection]

        # Iteratively refine cluster centers
        for iteration in range(self.max_iter):
            # Compute Euclidean distances from each embedding to each cluster center
            cluster_distances = spatial.distance.cdist(self.embeddings, self.cluster_centers, 'euclidean')
            nearest_cluster_indices = np.argmin(cluster_distances, axis=1)

            # Update cluster centers based on the mean of members in each cluster
            for cluster_index in range(self.num_clusters):
                cluster_members = self.embeddings[nearest_cluster_indices == cluster_index]
                if cluster_members.size > 0:
                    self.cluster_centers[cluster_index] = np.average(cluster_members, axis=0)

    # ToDo
    def predict(self, face):
        embedding = self.facenet.predict(face)
        distances = spatial.distance.cdist([embedding], self.cluster_center, 'euclidean')[0]
        # Picking the best cluster based on the minimum distance
        best = np.argmin(distances)
        return best, distances

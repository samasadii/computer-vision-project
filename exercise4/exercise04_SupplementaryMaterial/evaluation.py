import numpy as np
import pickle
from classifier import NearestNeighborClassifier

# Class label for unknown subjects in test and training data.
UNKNOWN_LABEL = -1


# Evaluation of open-set face identification.
class OpenSetEvaluation:

    def __init__(self,
                 classifier=NearestNeighborClassifier(),
                 false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True)):

        # The false alarm rates.
        self.false_alarm_rate_range = false_alarm_rate_range

        # Datasets (embeddings + labels) used for training and testing.
        self.train_embeddings = []
        self.train_labels = []
        self.test_embeddings = []
        self.test_labels = []

        # The evaluated classifier (see classifier.py)
        self.classifier = classifier

    # Prepare the evaluation by reading training and test data from file.
    def prepare_input_data(self, train_data_file, test_data_file):

        with open(train_data_file, 'rb') as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f)
        with open(test_data_file, 'rb') as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f)

    # Run the evaluation and find performance measure (identification rates) at different similarity thresholds.
    def run(self):

        # Fit the classifier on the training data
        self.classifier.fit(np.array(self.train_embeddings), np.array(self.train_labels))

        # Predict labels and similarities for the test data
        predicted_labels, similarities = self.classifier.predict_labels_and_similarities(np.array(self.test_embeddings))

        similarity_thresholds = []
        identification_rates = []

        # Iterate over each false alarm rate to find corresponding thresholds and identification rates
        for far in self.false_alarm_rate_range:
            threshold = self.select_similarity_threshold(similarities, far)
            similarity_thresholds.append(threshold)
            # Apply threshold to filter similarities and corresponding labels
            filtered_indices = similarities >= threshold
            filtered_labels = predicted_labels[filtered_indices]
            filtered_true_labels = np.array(self.test_labels)[filtered_indices]
            id_rate = self.calc_identification_rate(filtered_labels, filtered_true_labels)
            identification_rates.append(id_rate)

        # Report all performance measures.
        evaluation_results = {'similarity_thresholds': similarity_thresholds,
                              'identification_rates': identification_rates}

        return evaluation_results

    def select_similarity_threshold(self, similarity, false_alarm_rate):
        num_samples = len(similarity)
        threshold_index = int((1 - false_alarm_rate) * num_samples)
        sorted_similarities = np.sort(similarity)
        return sorted_similarities[max(0, threshold_index - 1)]

    def calc_identification_rate(self, prediction_labels, true_labels):
        if len(true_labels) == 0:
            return 0  # Avoid division by zero
        correct_identifications = prediction_labels == true_labels
        return np.sum(correct_identifications) / len(true_labels)

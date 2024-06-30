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
            (self.train_embeddings, self.train_labels) = pickle.load(f, encoding='latin1')
        with open(test_data_file, 'rb') as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f, encoding='latin1')

    # Run the evaluation and find performance measure (identification rates) at different similarity thresholds.
    def run(self):

        # Train the classifier using the training data
        self.classifier.fit(np.array(self.train_embeddings), np.array(self.train_labels))

        # Obtain predictions and similarity scores from the test dataset
        predicted_labels, similarity_scores = self.classifier.predict_labels_and_similarities(self.test_embeddings)

        thresholds = []
        rates_of_identification = []

        # Evaluate performance at various levels of false alarm rates
        for far in self.false_alarm_rate_range:
            cutoff = self.select_similarity_threshold(similarity_scores, far)
            filtered_labels = [label if score >= cutoff else UNKNOWN_LABEL for label, score in zip(predicted_labels, similarity_scores)]
            identification_rate = self.calc_identification_rate(filtered_labels, self.test_labels)
            thresholds.append(cutoff)
            rates_of_identification.append(identification_rate)

        return {'similarity_thresholds': thresholds, 'identification_rates': rates_of_identification}

    def select_similarity_threshold(self, similarities, false_alarm_rate):
        # Filter out the similarities of known subjects
        known_similarities = [s for s, label in zip(similarities, self.test_labels) if label != UNKNOWN_LABEL]
        # Calculate the similarity threshold for the given false alarm rate
        threshold = np.percentile(known_similarities, (1 - false_alarm_rate) * 100)
        return threshold

    def calc_identification_rate(self, prediction_labels, true_labels):
        # Calculate the identification rate based on the prediction labels and the true labels
        correct_predictions = np.sum(np.array(prediction_labels) == np.array(true_labels))
        total_predictions = len(true_labels)
        # Calculate the identification rate
        identification_rate = correct_predictions / total_predictions if total_predictions else 0
        return identification_rate

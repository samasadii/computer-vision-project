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

        with open(train_data_file, 'r') as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f)
        with open(test_data_file, 'r') as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f)

    # Run the evaluation and find performance measure (identification rates) at different similarity thresholds.
    def run(self):

        similarity_thresholds = None
	identification_rates = None

        # Report all performance measures.
        evaluation_results = {'similarity_thresholds': similarity_thresholds,
                              'identification_rates': identification_rates}

        return evaluation_results

    def select_similarity_threshold(self, similarity, false_alarm_rate):

        return None

    def calc_identification_rate(self, prediction_labels):

        return None

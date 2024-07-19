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
            (self.train_embeddings, self.train_labels) = pickle.load(f, encoding='iso-8859-1')
        with open(test_data_file, 'rb') as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f, encoding='iso-8859-1')

    # Run the evaluation and find performance measure (identification rates) at different similarity thresholds.
    def run(self):

        similarity_thresholds = []
        identification_rates = []

        self.classifier.fit(self.train_embeddings, self.train_labels)
        prediction_labels, similarities = self.classifier.predict_labels_and_similarities(self.test_embeddings)

        is_unknown = np.array(self.test_labels) == UNKNOWN_LABEL
        

        unknown_similarities = similarities[is_unknown]
        known_similarities = similarities[~is_unknown]
        known_prediction_labels = prediction_labels[~is_unknown]
        known_labels = np.array(self.test_labels)[~is_unknown]
        
        for far in self.false_alarm_rate_range:
            threshold = self.select_similarity_threshold(unknown_similarities, far)
            is_above_threshold = known_similarities >= threshold

            # Evaluate identification rate only on known samples
            valid_predictions = np.array(known_prediction_labels)[is_above_threshold]
            valid_labels = np.array(known_labels)[is_above_threshold]

            identification_rate = self.calc_identification_rate(valid_predictions, valid_labels, known_labels)

            similarity_thresholds.append(threshold)
            identification_rates.append(identification_rate)
        
        evaluation_results = {'similarity_thresholds': similarity_thresholds, 'identification_rates': identification_rates}


        return evaluation_results

    def select_similarity_threshold(self, similarities, false_alarm_rate):
        # Select the threshold that allows us to achieve the desired false alarm rate
        threshold = np.percentile(similarities, 100 * (1 - false_alarm_rate))
        return threshold

    def calc_identification_rate(self, prediction_labels, true_labels, known_labels):
        correct_predictions = np.sum(np.array(prediction_labels) == np.array(true_labels))
        total_predictions = len(known_labels)
        return correct_predictions / total_predictions

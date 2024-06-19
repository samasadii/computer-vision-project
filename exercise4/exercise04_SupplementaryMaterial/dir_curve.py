from evaluation import OpenSetEvaluation
from classifier import NearestNeighborClassifier
import matplotlib.pyplot as plt
import numpy as np

# The range of the false alarm rate in logarithmic space to draw DIR curves.
false_alarm_rate_range = np.logspace(-3.0, 0, 1000, endpoint=False)

# Pickle files containing embeddings and corresponding class labels for the training and the test dataset.
train_data_file = "evaluation_training_data.pkl"
test_data_file = "evaluation_test_data.pkl"

# We use a nearest neighbor classifier for this evaluation.
classifier = NearestNeighborClassifier()

# Prepare a new evaluation instance and feed training and test data into this evaluation.
evaluation = OpenSetEvaluation(classifier=classifier,
                               false_alarm_rate_range=false_alarm_rate_range)
evaluation.prepare_input_data(train_data_file, test_data_file)

# Run the evaluation and retrieve the performance measures (identification rates and false alarm rates) on the test
# dataset.
results = evaluation.run()

# Plot the DIR curve.
plt.semilogx(false_alarm_rate_range, results['identification_rates'],
             markeredgewidth=1,
             linewidth=3,
             linestyle="--",
             color="blue")
plt.grid(True)
plt.axis([false_alarm_rate_range[0], false_alarm_rate_range[len(false_alarm_rate_range)-1], 0, 1])
plt.xlabel('False alarm rate')
plt.ylabel('Identification rate')
plt.show()
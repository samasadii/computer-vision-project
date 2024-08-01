import pickle
from sklearn import svm
import joblib

def train_svm(features_path, model_path):
    with open(features_path, 'rb') as f:
        features, labels, pca = pickle.load(f)  # Load PCA with features and labels
    
    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(features, labels)

    # Save the trained SVM model
    joblib.dump(clf, model_path)

    # Save the PCA model
    pca_model_path = "results/pca_model.pkl"
    joblib.dump(pca, pca_model_path)

if __name__ == "__main__":
    train_svm("results/features_labels.pkl", "results/svm_model.pkl")

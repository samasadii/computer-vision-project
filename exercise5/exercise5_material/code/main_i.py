from proposal_generation import generate_proposals
from data_preprocessing import create_training_samples
from feature_extraction import extract_features
from svm_training import train_svm
from inference import run_inference
import os

# Paths
train_images_path = os.path.join("data", "train")
valid_images_path = os.path.join("data", "valid")
test_images_path = os.path.join("data", "test")
train_annotations_path = os.path.join("data", "train", "_annotations.coco.json")
valid_annotations_path = os.path.join("data", "valid", "_annotations.coco.json")
test_annotations_path = os.path.join("data", "test", "_annotations.coco.json")
proposals_path = os.path.join("results", "proposals.pkl")
model_path = os.path.join("results", "svm_model.pkl")

# Task 5.2.1: Generate proposals
generate_proposals([train_images_path, valid_images_path], proposals_path)

# Task 5.2.2: Create training samples
samples_path = create_training_samples(proposals_path, [train_annotations_path, valid_annotations_path], tp=0.75, tn=0.25)

# Task 5.2.3: Extract features
features_path = extract_features(samples_path)

# Task 5.2.4: Train SVM
train_svm(features_path, model_path)

# Task 5.2.5: Inference
run_inference(test_images_path, test_annotations_path, model_path)

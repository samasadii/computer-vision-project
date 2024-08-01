import json
import numpy as np
import pickle
from sklearn.metrics import jaccard_score
import os

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxB[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Denominator in IoU calculation
    denominator = boxAArea + boxBArea - interArea

    # Avoid division by zero
    if denominator == 0:
        return 0.0

    iou = interArea / float(denominator)
    return iou



def create_training_samples(proposals_file, annotations_files, tp=0.75, tn=0.25):
    with open(proposals_file, 'rb') as f:
        proposals = pickle.load(f)

    training_samples = []

    for annotations_file in annotations_files:
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)

        for img in annotations['images']:
            filename = img['file_name']
            if filename not in proposals:
                continue
            regions = proposals[filename]
            img_annotations = [anno for anno in annotations['annotations'] if anno['image_id'] == img['id']]
            gt_boxes = [anno['bbox'] for anno in img_annotations]
            gt_boxes = [ [x, y, x+w, y+h] for x, y, w, h in gt_boxes ] # Convert to [x1, y1, x2, y2]

            for region in regions:
                box = [region['rect'][0], region['rect'][1], region['rect'][2], region['rect'][3]]
                iou_scores = [calculate_iou(box, gt_box) for gt_box in gt_boxes]
                max_iou = max(iou_scores)

                if max_iou >= tp:
                    training_samples.append((filename, box, 'balloon'))
                elif max_iou <= tn:
                    training_samples.append((filename, box, 'background'))

    samples_path = "results/training_samples.pkl"
    with open(samples_path, 'wb') as f:
        pickle.dump(training_samples, f)
    
    return samples_path

if __name__ == "__main__":
    create_training_samples("results/proposals.pkl", ["data/train/annotations.coco.json", "data/valid/annotations.coco.json"])

import os
import pickle
from skimage.io import imread
from selective_search import selective_search

def generate_proposals(image_folders, output_file):
    # Check if the proposals file already exists
    if os.path.exists(output_file):
        print(f"Loading existing proposals from {output_file}")
        with open(output_file, 'rb') as f:
            proposals = pickle.load(f)
    else:
        print(f"Generating new proposals and saving to {output_file}")
        proposals = {}
        for folder in image_folders:
            for filename in os.listdir(folder):
                if filename.endswith(".jpg"):
                    image_path = os.path.join(folder, filename)
                    image = imread(image_path)
                    _, regions = selective_search(image, scale=500, sigma=0.9, min_size=10)
                    proposals[filename] = regions

        # Save the proposals to a file
        with open(output_file, 'wb') as f:
            pickle.dump(proposals, f)

    return proposals

if __name__ == "__main__":
    generate_proposals(["data/train", "data/valid"], "results/proposals.pkl")

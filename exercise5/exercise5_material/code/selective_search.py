'''
@author: Prathmesh R Madhu.
For educational purposes only
'''
# -*- coding: utf-8 -*-
from __future__ import division

import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy as np

def generate_segments(im_orig, scale, sigma, min_size):
    """
    Task 1: Segment smallest regions by the algorithm of Felzenswalb.
    1.1. Generate the initial image mask using felzenszwalb algorithm
    1.2. Merge the image mask to the image as a 4th channel
    """
    ### YOUR CODE HERE ###

    # Generate the initial image mask
    im_mask = skimage.segmentation.felzenszwalb(im_orig, scale=scale, sigma=sigma, min_size=min_size)

    # Merge the image mask to the image as a 4th channel
    im_orig = np.concatenate((im_orig, im_mask[:, : , np.newaxis]), axis=2)

    return im_orig

def sim_colour(r1, r2):
    """
    2.1. calculate the sum of histogram intersection of colour
    """
    ### YOUR CODE HERE ###

    sum = 0
    for a,b in zip(r1["colour_hist"], r2["colour_hist"]):
        sum += min(a, b)
    return sum

def sim_texture(r1, r2):
    """
    2.2. calculate the sum of histogram intersection of texture
    """
    ### YOUR CODE HERE ###

    sum = 0
    for a, b in zip(r1["texture_hist"], r2["texture_hist"]):
        sum += min(a, b)
    return sum


def sim_size(r1, r2, imsize):
    """
    2.3. calculate the size similarity over the image
    """
    ### YOUR CODE HERE ###

    return 1.0 - ((r1["size"] + r2["size"]) / imsize)


def sim_fill(r1, r2, imsize):
    """
    2.4. calculate the fill similarity over the image
    """
    ### YOUR CODE HERE ###

    # Calculate the size of the combined bounding box by multiplying the width and height of the bounding box
    bounding_box_size = (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"])) * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))

    # Calculate the fill similarity
    fill_similarity = 1.0 - ((bounding_box_size - r1["size"] - r2["size"]) / imsize)

    return fill_similarity

def calc_sim(r1, r2, imsize):
    return (sim_colour(r1, r2) + sim_texture(r1, r2)
            + sim_size(r1, r2, imsize) + sim_fill(r1, r2, imsize))

def calc_colour_hist(img):
    """
    Task 2.5.1
    calculate colour histogram for each region
    the size of output histogram will be BINS * COLOUR_CHANNELS(3)
    number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
    extract HSV
    """
    BINS = 25
    hist = np.array([])
    
    ### YOUR CODE HERE ###

    # Calculate the color histogram for each channel
    histograms = [np.histogram(img[:, channel], BINS, (0.0, 255.0))[0] for channel in range(3)]
    
    # Concatenate the histograms
    hist = np.concatenate(histograms)

    # L1 normalize the histogram
    hist = hist / (img.shape[0] * img.shape[1])

    return hist

def calc_texture_gradient(img):
    """
    Task 2.5.2
    calculate texture gradient for entire image
    The original SelectiveSearch algorithm proposed Gaussian derivative
    for 8 orientations, but we will use LBP instead.
    output will be [height(*)][width(*)]
    Useful function: Refer to skimage.feature.local_binary_pattern documentation
    """
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    ### YOUR CODE HERE ###

    # Calculate the texture gradient for each channel
    ret = np.stack([skimage.feature.local_binary_pattern(img[:, :, channel], 8, 1.0) for channel in range(3)], axis=-1)

    return ret

def calc_texture_hist(img):
    """
    Task 2.5.3
    calculate texture histogram for each region
    calculate the histogram of gradient for each colours
    the size of output histogram will be
        BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    Do not forget to L1 Normalize the histogram
    """
    BINS = 10
    hist = np.array([])
    ### YOUR CODE HERE ###

    # Calculate the texture histogram for each channel
    histograms = [np.histogram(img[:, channel], BINS, (0.0, 1.0))[0] for channel in range(3)]

    # Concatenate the histograms
    hist = np.concatenate(histograms)

    # L1 Normalize the histogram
    hist = hist / (img.shape[0] * img.shape[1])

    return hist

def extract_regions(img):
    '''
    Task 2.5: Generate regions denoted as datastructure R
    - Convert image to hsv color map
    - Count pixel positions
    - Calculate the texture gradient
    - calculate color and texture histograms
    - Store all the necessary values in R.
    '''
    R = {}
    ### YOUR CODE HERE ###

    # Convert the image to the HSV color space
    hsv_image = skimage.color.rgb2hsv(img[:, :, :3])

    # Iterate each pixel in the image
    for y, row in enumerate(img):
        for x, (r, g, b, l) in enumerate(row):
            if l not in R:
                # Add the region to the dictionary if it does not exist
                R[l] = {"min_x": 0xffff, "min_y": 0xffff, "max_x": 0, "max_y": 0, "labels": [l] }

            # Update the region's bounding box
            R[l]["min_x"] = min(R[l]["min_x"], x)
            R[l]["min_y"] = min(R[l]["min_y"], y)
            R[l]["max_x"] = max(R[l]["max_x"], x)
            R[l]["max_y"] = max(R[l]["max_y"], y)

    # Calculate the texture gradient
    texture_gradient = calc_texture_gradient(img)

    # Iterate through each region in the image
    for k, v in R.items():
        # Mask the pixels in the region
        masked_pixels = hsv_image[img[:, :, 3] == k]
        # Calculate the size, color histogram, and texture histogram of the region
        R[k]["size"] = len(masked_pixels / 4)
        R[k]["colour_hist"] = calc_colour_hist(masked_pixels)
        R[k]["texture_hist"] = calc_texture_hist(texture_gradient[img[:, :, 3] == k])

    return R

def extract_neighbours(regions):

    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    # Hint 1: List of neighbouring regions
    # Hint 2: The function intersect has been written for you and is required to check neighbours
    neighbours = []
    ### YOUR CODE HERE ###

    keys = list(regions.keys())

    # Iterate through each pair of regions
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            # Check if the regions intersect
            label1, label2 = keys[i], keys[j]
            region1, region2 = regions[label1], regions[label2]
            if intersect(region1, region2):
                # Add the pair of regions to the list of neighbours
                neighbours.append(((label1, region1), (label2, region2)))

    return neighbours

def merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {}
    ### YOUR CODE HERE
        
    # Calculate the new bounding box, size, color histogram, texture histogram, and labels of the merged region
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "colour_hist": (r1["colour_hist"] * r1["size"] + r2["colour_hist"] * r2["size"]) / new_size,
        "texture_hist": (r1["texture_hist"] * r1["size"] + r2["texture_hist"] * r2["size"]) / new_size,
        "labels": r1["labels"] + r2["labels"]
    }

    return rt


def selective_search(image_orig, scale=1.0, sigma=0.8, min_size=50):
    '''
    Selective Search for Object Recognition" by J.R.R. Uijlings et al.
    :arg:
        image_orig: np.ndarray, Input image
        scale: int, determines the cluster size in felzenszwalb segmentation
        sigma: float, width of Gaussian kernel for felzenszwalb segmentation
        min_size: int, minimum component size for felzenszwalb segmentation

    :return:
        image: np.ndarray,
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions: array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''

    # Checking the 3 channel of input image
    assert image_orig.shape[2] == 3, "Please use image with three channels."
    imsize = image_orig.shape[0] * image_orig.shape[1]

    # Task 1: Load image and get smallest regions. Refer to `generate_segments` function.
    image = generate_segments(image_orig, scale, sigma, min_size)

    if image is None:
        return None, {}

    # Task 2: Extracting regions from image
    # Task 2.1-2.4: Refer to functions "sim_colour", "sim_texture", "sim_size", "sim_fill"
    # Task 2.5: Refer to function "extract_regions". You would also need to fill "calc_colour_hist",
    # "calc_texture_hist" and "calc_texture_gradient" in order to finish task 2.5.
    R = extract_regions(image)

    # Task 3: Extracting neighbouring information
    # Refer to function "extract_neighbours"
    neighbours = extract_neighbours(R)

    # Calculating initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = calc_sim(ar, br, imsize)

    # Hierarchical search for merging similar regions
    while S != {}:

        # Get highest similarity
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # Task 4: Merge corresponding regions. Refer to function "merge_regions"
        t = max(R.keys()) + 1.0
        R[t] = merge_regions(R[i], R[j])

        # Task 5: Mark similarities for regions to be removed
        ### YOUR CODE HERE ###

        # Get the list of regions' keys that are similar to the merged regions
        similar_regions = [key for key, _ in S.items() if i in key or j in key]

        # Task 6: Remove old similarities of related regions
        ### YOUR CODE HERE ###

        # Remove the old similarities of the related regions
        for key in similar_regions:
            del S[key]

        # Task 7: Calculate similarities with the new region
        ### YOUR CODE HERE ###

        keys = [key for key in similar_regions if key != (i, j)]
        for key in keys:
            if(key[0] in (i, j)):
                n = key[1]
            else:
                n = key[0]
            S[(t, n)] = calc_sim(R[t], R[n], imsize)


    # Task 8: Generating the final regions from R
    regions = []
    ### YOUR CODE HERE ###
    for _, r in R.items():
        regions.append({
            'rect': (r['min_x'], r['min_y'], r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
            'size': r['size'],
            'labels': r['labels']
        })


    return image, regions



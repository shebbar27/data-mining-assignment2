import collections
import cv2
import numpy as np
from sklearn.cluster import DBSCAN


SATURATION_THRESH = 72
BLACK_COLOR = (0, 0, 0)
YELLOW_COLOR = (0, 233, 255)
WHITE_PIXEL = 255
BLACK_PIXEL = 0
BINARY_THRESHOLD = 32
EPS = 2
MIN_SAMPLES = 2


def get_feature_points(binary_image):
    featurePoints = []
    rows, cols = binary_image.shape
    for row in range(rows):
        for col in range(cols):
            if (binary_image[row, col] > BLACK_PIXEL):
                featurePoints.append([row, col])
    return np.array(featurePoints);


def count_clusters(clusters_image, min_pixels_in_cluster):
    bgr_image = cv2.cvtColor(clusters_image, cv2.COLOR_HSV2BGR)
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, BINARY_THRESHOLD, WHITE_PIXEL, cv2.THRESH_BINARY)
    feature_points = get_feature_points(binary_image)
    label_counts = {}
    if (len(feature_points) != 0):
        dbscan_model = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
        clusters = dbscan_model.fit(feature_points)
        label_counts = collections.Counter(clusters.labels_)

    num_of_significant_clusters = 0
    for (label, count) in label_counts.items():
        if label != -1 and count >= min_pixels_in_cluster:
            num_of_significant_clusters+=1
    return num_of_significant_clusters


def find_clusters(brain_image, min_pixels_in_cluster):
    hsv = cv2.cvtColor(brain_image, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)

    mask = cv2.threshold(s, SATURATION_THRESH, WHITE_PIXEL, cv2.THRESH_BINARY)[1]
    clusters_image = brain_image.copy()
    clusters_image[mask==0] = BLACK_COLOR
    clusters_image[mask>0] = YELLOW_COLOR

    clusters_count = count_clusters(clusters_image, min_pixels_in_cluster)
    return clusters_image, clusters_count
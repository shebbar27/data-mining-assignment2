import clustering
import csv
import cv2
import os
import shutil


INPUT_DIR = 'testPatient/'
SLICES_OUTPUT_DIR = 'Slices/'
CLUSTERS_OUTPUT_DIR = 'Clusters/'
IMAGE_FILE_SUFFIX = 'thresh.png'
IMAGE_EXTENSION = '.png'
MIN_PIXELS_IN_CLUSTER = 135
SLICES_CLUSTER_COUNT_FIELDS = ['SliceNumber', 'ClusterCount']


# utility function to remove file extension form file name
def remove_file_extension(file_name):
    return os.path.splitext(file_name)[0]


# utility function to join directory path with file name
def join_path(dir, filename):
    return os.path.join(dir, filename)


# utlity function to clear all contents of output directory
def init_output_dirs(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)


# utility function to read all the input images form the testPatient directory
def read_input_data():
    images = []
    for file_name in os.listdir(INPUT_DIR):
        if(file_name.endswith(IMAGE_FILE_SUFFIX)):
            brain_image = cv2.imread(join_path(INPUT_DIR, file_name))
            images.append((file_name, brain_image))
    return images


def write_to_scv_file(file_path, header, rows):
    with open(file_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


# utiltiy function to check whether given two rectangles overlap with each other 
def is_overlapping(rect1, rect2):
    dx = min(rect1[2], rect2[2]) - max(rect1[0], rect2[0])
    dy = min(rect1[3], rect2[3]) - max(rect1[1], rect2[1])

    if dx > 0 and dy > 0:
        return True
    return False


# verifify whether the boundary coordinates are valid and add them valid boundary coordinates
def validate_coordinates(x, y, w, h, valid_slice_coordinates, invalid_slice_coordinates, offset_pixels):
    MIN_DIMENSION = 10
    MAX_DIMENSION = 250
    # reject if width or height is less than 10 pixels
    if w < MIN_DIMENSION or h < MIN_DIMENSION:
        return
    
    # reject if width or height is greater than 200 pixels
    if w > MAX_DIMENSION or h > MAX_DIMENSION:
        return
    
    # assume first boundary coordinates as valid
    if len(valid_slice_coordinates) == 0:
        valid_slice_coordinates.add((x, y, w, h))
        return

    is_valid = True
    # for each existing valid boundary coordinates check for overlap with new coordinates
    for x1, y1, w1, h1 in valid_slice_coordinates:
        if is_overlapping([x - offset_pixels, y - offset_pixels, x + w + offset_pixels, y + h + offset_pixels], [x1, y1, x1 + w1, y1 + h1]):
            if w * h > w1 * h1:
                invalid_slice_coordinates.add((x1, y1, w1, h1))
            else:
                is_valid = False
                    
    if is_valid:
        valid_slice_coordinates.add((x, y, w, h))


# function to slice input fMRI image containing multipe brain images into individual images
def slice_brain_image(brain_image):
    BINARY_THRESHOLD = 65
    MAX_PIXEL_VALUE = 255
    OFFSET_PIXELS = 15

    brain_images = []
    valid_slice_coordinates = set()
    invalid_slice_coordinates = set()
    gray_image = cv2.cvtColor(brain_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, BINARY_THRESHOLD, MAX_PIXEL_VALUE, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        validate_coordinates(x, y, w, h, valid_slice_coordinates, invalid_slice_coordinates, OFFSET_PIXELS)

    for coordinate in invalid_slice_coordinates:
        valid_slice_coordinates.discard(coordinate)

    for x, y, w, h in valid_slice_coordinates:
        # cv2.rectangle(brain_image, (x - OFFSET_PIXELS, y - OFFSET_PIXELS), (x + w + OFFSET_PIXELS, y + h + OFFSET_PIXELS), (0, 0, 255), 1)
        slice = brain_image.copy()[y - OFFSET_PIXELS : y + h + OFFSET_PIXELS, x - OFFSET_PIXELS :  x + w + OFFSET_PIXELS]
        brain_images.append(slice)

    # brain_images.append(brain_image)
    return brain_images


def main():
    # initialize Slices output directory
    init_output_dirs(SLICES_OUTPUT_DIR)
    
    # read all the required input images from 'testPatient' input directory
    images = read_input_data()
    
    # extract slices from each input image and store the slices in the Slices output directory
    for file_name, image in images:
        slices_dir = join_path(SLICES_OUTPUT_DIR, remove_file_extension(file_name))
        os.makedirs(slices_dir, exist_ok = True)
        slices = slice_brain_image(image)
        index = 1
        for slice in slices:
            cv2.imwrite(join_path(slices_dir, str(index) + IMAGE_EXTENSION), slice)
            index += 1

    # initialise Boundaries output directory
    init_output_dirs(CLUSTERS_OUTPUT_DIR)

    # get list of all sub directories under Slices folder    
    slice_dirs = [dir for dir in os.listdir(SLICES_OUTPUT_DIR) if os.path.isdir(join_path(SLICES_OUTPUT_DIR, dir))]

    # for each sub directory under Slices directory read all the slice image files and 
    # find clusters for each of the slices and write the output images to Clusters directory
    # also create a csv file listing clusters count for each slice
    for slice_dir in slice_dirs:
        source_path = join_path(SLICES_OUTPUT_DIR, slice_dir)
        sub_directory = join_path(CLUSTERS_OUTPUT_DIR, slice_dir)
        os.makedirs(sub_directory, exist_ok = True)
        slices_clusters_count_list = []
        for file_name in os.listdir(source_path):
            brain_image = cv2.imread(join_path(source_path, file_name))
            cluster_image, clusters_count = clustering.find_clusters(brain_image, MIN_PIXELS_IN_CLUSTER)
            cv2.imwrite(join_path(sub_directory, file_name), cluster_image)
            slices_clusters_count_list.append([remove_file_extension(file_name), clusters_count])

        slices_clusters_count_file = f'{join_path(sub_directory, slice_dir)}.csv'
        write_to_scv_file(
            slices_clusters_count_file, 
            SLICES_CLUSTER_COUNT_FIELDS, 
            slices_clusters_count_list)


if __name__ == '__main__':
    main()
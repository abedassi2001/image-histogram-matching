# Student_Name1, Student_ID1
# Student_Name2, Student_ID2

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")

# Input: numpy array of images and number of gray levels to quantize the images down to
# Output: numpy array of images, each with only n_colors gray levels
def quantization(imgs_arr, n_colors=4):
    img_size = imgs_arr[0].shape
    res = []

    for img in imgs_arr:
        X = img.reshape(img_size[0] * img_size[1], 1)
        km = KMeans(n_clusters=n_colors)
        km.fit(X)

        img_compressed = km.cluster_centers_[km.labels_]
        img_compressed = np.clip(img_compressed.astype('uint8'), 0, 255)

        res.append(img_compressed.reshape(img_size[0], img_size[1]))

    return np.array(res)

# Input: A path to a folder and formats of images to read
# Output: numpy array of grayscale versions of images read from input folder, and also a list of their names
def read_dir(folder, formats=(".jpg", ".png")):
    image_arrays = []
    lst = [file for file in os.listdir(folder) if file.endswith(formats)]
    for filename in lst:
        file_path = os.path.join(folder, filename)
        image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_arrays.append(gray_image)
    return np.array(image_arrays), lst

# Input: image object (as numpy array) and the index of the wanted bin (between 0 to 9)
# Output: the height of the idx bin in pixels
def get_bar_height(image, idx):
    # Assuming the image is of the same pixel proportions as images supplied in this exercise, the following values will work
    x_pos = 70 + 40 * idx
    y_pos = 274
    while image[y_pos, x_pos] == 1:
        y_pos-=1
    return 274 - y_pos

# Sections c, d
# Remember to uncomment compare_hist before using it!

def compare_hist(src_image, target):
    # Get sliding windows from the source image
    src_windows = np.lib.stride_tricks.sliding_window_view(src_image, target.shape)
    # Calculate the histogram and cumulative histogram of the target
    target_hist = cv2.calcHist([target], [0], None, [256], [0, 256]).flatten()
    target_cumsum_hist = np.cumsum(target_hist)
    for x in range(src_windows.shape[0]):
        for y in range(src_windows.shape[1]):

            # Extract the current window
            window = src_windows[x,y]

            # Calculate the histogram and cumulative histogram of the current window
            window_hist = cv2.calcHist([window], [0], None, [256], [0, 256]).flatten()
            window_cumsum_hist = np.cumsum(window_hist)

            # Compute the Earth Mover's Distance (EMD)
            emd = abs(np.subtract(window_cumsum_hist, target_cumsum_hist))
            emd_sum = np.sum(emd)
            threshold = 200
            if emd_sum <= threshold:
                window = src_windows[x, y]
                cv2.imshow("Sliding Window", window)
                cv2.waitKey(200)  # Wait 200ms to view each window
                cv2.destroyAllWindows()
                return True

    # If no match is found
    return False

def get_highest_digit(src_image , number_image):
    for i in range(8,0):
        if compare_hist(src_image, number_image[i]):
            return i
    return -1
# Sections a, b

images, names = read_dir('data')
numbers, numbers_names = read_dir('numbers')
print(compare_hist(images[0], numbers[0]))
print(compare_hist(images[0], numbers[1]))
print(compare_hist(images[0], numbers[2]))
print(compare_hist(images[0], numbers[3]))
print(compare_hist(images[0], numbers[4]))
print(compare_hist(images[0], numbers[5]))
print(compare_hist(images[0], numbers[6]))
print(compare_hist(images[0], numbers[7]))
print(compare_hist(images[0], numbers[8]))
print(compare_hist(images[0], numbers[9]))
#quantize the images
quantized_images = quantization(images, 5)

for i in range (0,7):
    cv2.imshow(names[i], quantized_images[i])

int_list = [0] * 10
for i in range(0,10):
    int_list[i] = get_bar_height(quantized_images[i], i)
cv2.imshow(names[0],images[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
#print(len(images[0]))
#print(numbers[6].shape)
# Get the image dimensions (width, height)
cv2.waitKey(0)
cv2.destroyAllWindows() 
exit()


# The following print line is what you should use when printing out the final result - the text version of each histogram, basically.

# print(f'Histogram {names[id]} gave {heights}')

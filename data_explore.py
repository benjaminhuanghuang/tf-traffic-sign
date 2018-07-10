import os
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
from skimage import transform
from skimage import data
import numpy as np

from utils import load_data, display_images_and_labels

# Load training and testing datasets.
ROOT_PATH = "./data"
train_data_dir = os.path.join(ROOT_PATH, "Training")
test_data_dir = os.path.join(ROOT_PATH, "Testing")

# get images and labels (0 to 61) as list
images, labels = load_data(train_data_dir)
# convert the variables to array
images_array = np.array(images)
labels_array = np.array(labels)

# Print the `images` dimensions
print(images_array.ndim)
# Print the number of `images`'s elements
print(images_array.size)
# Print the first instance of `images`
images_array[0]

# Print the `labels` dimensions
print(labels_array.ndim)
# Print the number of `labels`'s elements
print(labels_array.size)
# Count the number of labels
print(len(set(labels_array)))

# ----------------------------------------------------
# Make a histogram with 62 bins of the `labels` data
plt.hist(labels, 62)
# Show the plot
plt.show()

# Determine the (random) indexes of the images that you want to see 
traffic_signs = [300, 2250, 3650, 4000]

# Fill out the subplots with the random images that you defined 
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)

plt.show()

display_images_and_labels(images, labels)

# Adjust the image size
images32 = [transform.resize(image, (32, 32)) for image in images]
images32 = np.array(images32)
display_images_and_labels(images32, labels)



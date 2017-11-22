# import tensorflow to the environment
import tensorflow as tf
from PIL import Image
from skimage import transform
from skimage import data
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.color import rgb2gray
import random


# Loading And Exploring The Data
def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".JPG")]
        for f in file_names:
            images.append(data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "/home/rose/Documents/"
train_data_directory = os.path.join(ROOT_PATH, "MazeProject/Training")
test_data_directory = os.path.join(ROOT_PATH, "MazeProject/Testing")

images, labels = load_data(train_data_directory)

images_array = np.array(images)
labels_array = np.array(labels)

# Print the `images` dimensions
#print(images_array.ndim)

# Print the number of `images`'s elements
#print(images_array.size)

# Print the first instance of `images`
#print(images[0])

# Print the `labels` dimensions (label is a subdirectory)
#print(labels_array.ndim)

# Print the number of `labels`'s elements
#print(labels_array.size)

# Count the number of labels
#print(len(set(labels_array)))

# Make a histogram with 3 bins of the `labels` data
#plt.hist(labels, 3)

# Show the plot
#plt.show()

# Determine the (random) indexes of the images
maze = [3, 15, 18, 25]

# Get the unique labels
#unique_labels = set(labels)

# Initialize the figure
#plt.figure(figsize=(15, 15))

# Set a counter
#i = 1

# For each unique label,
#for label in unique_labels:
    # You pick the first image for each label
#    image = images[labels.index(label)]
    # Define 64 subplots
#    plt.subplot(8, 8, i)
    # Don't include axes
#    plt.axis('off')
    # Add a title to each subplot
#    plt.title("Label {0} ({1})".format(label, labels.count(label)))
    # Add 1 to the counter
#    i += 1
    # And you plot this first image
    #plt.imshow(image)

# Show the plot
#plt.show()

# Resize images
images32 = [transform.resize(image, (30, 30)) for image in images]
images32 = np.array(images32)
images32 = rgb2gray(np.array(images32))

for i in range(len(maze)):
    plt.subplot(1, 4, i + 1)
    plt.axis('off')
    plt.imshow(images32[maze[i]], cmap="gray")
    plt.subplots_adjust(wspace=0.5)
#plt.show()
#print(images32.shape)

x = tf.placeholder(dtype = tf.float32, shape = [None, 30, 30])
y = tf.placeholder(dtype = tf.int32, shape = [None])
images_flat = tf.contrib.layers.flatten(x)
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_pred = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#print("images_flat: ", images_flat)
#print("logits: ", logits)
#print("loss: ", loss)
#print("predicted_labels: ", correct_pred)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(201):
#        print('EPOCH', i)
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images32, y: labels})
        if i % 10 == 0:
            loss
#            print("Loss: ", loss)
#        print('DONE WITH EPOCH')

test_images, test_labels = load_data(test_data_directory)

# Pick 10 random images
sample_indexes = random.sample(range(len(images32)), 10)
sample_images = [images32[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Run the "predicted_labels" op.
predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]

# Print the real and predicted labels
print(sample_labels)
print(predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
             fontsize=12, color=color)
    plt.imshow(sample_images[i])

plt.show()

# Calculate correct matches
match_count = sum([int(y == y_) for y, y_ in zip(sample_labels, predicted)])

# Calculate the accuracy
accuracy = match_count / len(sample_labels)

# Print the accuracy
print("Accuracy: {:.3f}".format(accuracy))

sess.close()
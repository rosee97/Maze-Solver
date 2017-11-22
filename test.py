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

'''
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
#train_data_directory = os.path.join(ROOT_PATH, "MazeProject/Training")
test_data_directory = os.path.join(ROOT_PATH, "MazeProject/Testing")
'''


import cv2

img = cv2.imread('/home/rose/Documents/MazeProject/0.JPG')
img = cv2.resize(img,(100, 100))
img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
image_data = np.asarray(img)

images, labels = [image_data],  [1]  #   load_data(test_data_directory)







# Resize images
images32 = [transform.resize(image, (30, 30)) for image in images]
images32 = np.array(images32)
images32 = rgb2gray(np.array(images32))

x = tf.placeholder(dtype = tf.float32, shape = [None, 30])
y = tf.placeholder(dtype = tf.int32, shape = [None])
images_flat = tf.contrib.layers.flatten(x)
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_pred = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.set_random_seed(1234)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(201):
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images32, y: labels})
        if i % 10 == 0:
            loss


#train_images, train_labels = load_data(train_data_directory)
#test_images, test_labels = load_data(test_data_directory)

#print(test_images)

# Transform the images to 30 by 30 pixels
images_array30 = [transform.resize(image, (30, 30)) for image in images]
#test_images30 = [transform.resize(image, (30, 30)) for image in test_images]

# Convert to grayscale
images_array30 = rgb2gray(np.array(images_array30))
#test_images30 = rgb2gray(np.array(test_images30))

# Run predictions against the full test set.
train_predicted = sess.run([correct_pred], feed_dict={x: images_array30})[0]
#predicted = sess.run([correct_pred], feed_dict={x: test_images30})[0]

fig = plt.figure(figsize=(10, 10))
for i in range(len(images)):
    truth = labels[i]
    prediction = train_predicted[i]
    plt.subplot(10/2, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(20, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
             fontsize=10, color=color)
    plt.imshow(images[i])

plt.show()

print(labels)
print(train_predicted)

# Calculate correct matches
match_count = sum([int(y == y_) for y, y_ in zip(labels, train_predicted)])

# Calculate the accuracy
accuracy = match_count / len(labels)

# Print the accuracy
print("Accuracy: {:.3f}".format(accuracy))

#Create a saver object which will save all the variables
saver = tf.train.Saver()

save_path = saver.save(sess, '/home/rose/Documents/MazeProject/train_data1.ckpt')

saver.restore(sess, save_path)

sess.close()

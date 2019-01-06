from load_dataset import loadDataset
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

train_data = loadDataset(0.75, 0.1, 0.15, "./flowers-recognition/flowers/*/*.jpg", "train_resize.tfrecords", "test_resize.tfrecords", "valid_resize.tfrecords")
train_data.createDataRecordAll()

train_images, train_labels, train_fns = train_data.getTrainBatches("train_resize.tfrecords", batch_size=4, isTrain=True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    image_data, label_data, files = sess.run([train_images, train_labels, train_fns])
    
original = mpimg.imread(files[0], format='jpg')
plt.imshow(original)

retrived_image = np.array(image_data[0], dtype=np.uint8)
plt.imshow(retrived_image)
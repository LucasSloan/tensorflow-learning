import tensorflow as tf
import save_images
import numpy as np

cifar_categories = ["airplane", "automobile", "bird",
                    "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

CHANNELS = 3


def parse_images(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=CHANNELS)
    image_float = tf.image.convert_image_dtype(image_decoded, tf.float32)
    return image_float


def text_to_one_hot(text_label):
    int_label = tf.string_to_number(text_label, out_type=tf.int32)
    return tf.one_hot(int_label, 10)


def text_to_index(text_label):
    return tf.string_to_number(text_label, out_type=tf.int32)


def load_images_and_labels(batch_size, image_glob, label_file):
    image_files_dataset = tf.data.Dataset.list_files(image_glob, shuffle=False)
    image_dataset = image_files_dataset.map(parse_images, num_parallel_calls=8)

    label_lines_dataset = tf.data.TextLineDataset(label_file)
    label_dataset = label_lines_dataset.map(text_to_one_hot)
    index_dataset = label_lines_dataset.map(text_to_index)

    dataset = tf.data.Dataset.zip(
        (image_dataset, label_dataset, index_dataset))

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    return dataset


training_dataset = load_images_and_labels(
    100, "E:\\cifar10\\train\\*", "E:\\cifar10\\Train_cntk_text.txt")
testing_dataset = load_images_and_labels(
    100, "E:\\cifar10\\test\\*", "E:\\cifar10\\Test_cntk_text.txt")

iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)
next_element = iterator.get_next()
x = next_element[0]
y_ = next_element[1]
indexes = next_element[2]

training_init_op = iterator.make_initializer(training_dataset)
testing_init_op = iterator.make_initializer(testing_dataset)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 32x32x1 -> 16x16x32
W_conv1 = weight_variable([5, 5, CHANNELS, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 32, 32, CHANNELS])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 16x16x32 -> 8x8x64
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([8 * 8 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(30):
        sess.run(training_init_op)
        i = 0
        while True:
            try:
                i += 1
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(
                        feed_dict={keep_prob: 1.0})
                    print('step %d, training accuracy %g' %
                          (i, train_accuracy))
                train_step.run(feed_dict={keep_prob: 0.5})
            except tf.errors.OutOfRangeError:
                break
        sess.run(testing_init_op)
        test_accuracy = sess.run(accuracy, feed_dict={keep_prob: 1.0})
        print('accuracy after {} epochs: {:.4f}'.format(epoch, test_accuracy))

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
    image_flipped = tf.image.flip_left_right(image_float)
    return image_flipped


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


def residual_block(input, channels, downsample):
    strides = (1, 1)
    if downsample:
        strides = (2, 2)
        
    conv1 = tf.layers.conv2d(input, channels, (3, 3), strides=strides, padding='same')
    conv1 = tf.layers.batch_normalization(conv1, training=True)

    conv2 = tf.layers.conv2d(conv1, channels, (3, 3), strides=(1, 1), padding='same')
    conv2 = tf.layers.batch_normalization(conv2, training=True)

    shortcut = tf.layers.conv2d(input, channels, (1, 1), strides=strides, padding='same')
    shortcut = tf.layers.batch_normalization(shortcut, training=True)

    output = shortcut + conv2
    output = tf.nn.leaky_relu(output)

    return output

x_image = tf.reshape(x, [-1, 32, 32, 3])

# 32x32x3 -> 16x16x32
res1 = residual_block(x_image, 32, True)

# 16x16x32 -> 8x8x64
res2 = residual_block(res1, 64, True)

# 8x8x64 -> 4x4x128
res3 = residual_block(res2, 128, True)

res3_flat = tf.reshape(res3, [-1, 2*2*256])

linear1 = tf.layers.dense(res3_flat, 1024, tf.nn.leaky_relu)
keep_prob = tf.placeholder(tf.float32)
linear1_drop = tf.nn.dropout(linear1, keep_prob)

linear2 = tf.layers.dense(linear1_drop, 10)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=linear2))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(linear2, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1, 11):
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

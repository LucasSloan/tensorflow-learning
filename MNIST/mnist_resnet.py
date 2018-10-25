from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])


def residual_block(input, channels):
    conv1 = tf.layers.conv2d(input, channels, (3, 3), strides=(2, 2), padding='same')
    conv1 = tf.layers.batch_normalization(conv1, training=True)

    conv2 = tf.layers.conv2d(conv1, channels, (3, 3), strides=(1, 1), padding='same')
    conv2 = tf.layers.batch_normalization(conv2, training=True)

    shortcut = tf.layers.conv2d(input, channels, (1, 1), strides=(2, 2), padding='same')
    shortcut = tf.layers.batch_normalization(shortcut, training=True)

    output = shortcut + conv2
    output = tf.nn.leaky_relu(output)

    return output

x_image = tf.reshape(x, [-1, 28, 28, 1])

# 28*28*1 -> 14*14*32
res1 = residual_block(x_image, 32)

# 14*14*32 -> 7*7*64
res2 = residual_block(res1, 64)

res2_flat = tf.reshape(res2, [-1, 7*7*64])

linear1 = tf.layers.dense(res2_flat, 1024, tf.nn.leaky_relu)
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
    for i in range(1, 20001):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(
                feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        if i % 1000 == 0:
            print('test accuracy %4f' % accuracy.eval(
                feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import dataload
import utils
from PIL import Image

data_dict = None
vgg_npy_path = "weights/vgg19.npy"
DATA_SET_PATH = 'data/nyu_depth_v2_training.mat'
TEST_DATA_SET_PATH = 'data/nyu_depth_v2_test.mat'
is_training = False
TRAIN_STEP = 100000
BATCH_SIZE = 16

Height = 160
Width = 120

Out_Height = 80
Out_Width = 60

lr = 0.0001

def avg_pool(input, name):
    return tf.nn.avg_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def max_pool(input, name):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def conv_filter(in_filters, out_filters):
    init = tf.truncated_normal(shape=[3, 3, in_filters, out_filters], dtype="float32")
    return tf.Variable(init)

def conv_bias(out_filters):
    init = tf.constant(0.1, dtype=tf.float32, shape=[out_filters])
    return tf.Variable(init, dtype="float32")

def get_conv_filter( name):
    init = tf.constant(data_dict[name][0])
    return tf.Variable(init, dtype="float32")


def get_conv_bias(name):
    init = tf.constant(data_dict[name][1])
    return tf.Variable(init, dtype="float32")

def conv_layer(input, name):
    filter = get_conv_filter(name)
    conv = tf.nn.conv2d(input, filter, [1, 1, 1, 1], padding='SAME')
    bias = get_conv_bias(name)
    return tf.nn.leaky_relu(tf.nn.bias_add(conv, bias))


def deconv_layer(input, in_filters, out_filters, flag):
    shape = tf.shape(input)
    filter = conv_filter(out_filters, in_filters)
    deconv = tf.nn.conv2d_transpose(input, filter, [BATCH_SIZE, shape[1]*2-flag, shape[2]*2, out_filters], [1, 2, 2, 1], padding='SAME')
    bias = conv_bias(out_filters)
    x = tf.nn.bias_add(deconv, bias)
    bn = batch_norm(x, is_training=is_training)
    return tf.nn.leaky_relu(bn)


def conv_layer2(input, in_filters, out_filters, name=None):
    filter = conv_filter(in_filters, out_filters)
    conv = tf.nn.conv2d(input, filter,[1, 1, 1, 1], padding='SAME' )
    bias = conv_bias(out_filters)
    return tf.nn.leaky_relu(tf.nn.bias_add(conv, bias))

def batch_norm(inputs, is_training,  epsilon = 0.001, decay = 0.99):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)

data_dict = np.load(vgg_npy_path, encoding='latin1').item()

x = tf.placeholder(dtype="float32", shape=[None,Width, Height,3], name="input")
y = tf.placeholder(dtype="float32", shape=[None,Out_Width,Out_Height], name="depth")

y_reshape = tf.reshape(y, [-1, Out_Width*Out_Height])
# convolution
# 160*120 64
conv1_1 = conv_layer(x, "conv1_1")
conv1_2 = conv_layer(conv1_1, "conv1_2")
pool1 = max_pool(conv1_2, 'pool1')
# 80*60 128
conv2_1 = conv_layer(pool1, "conv2_1")
conv2_2 = conv_layer(conv2_1, "conv2_2")
pool2 = max_pool(conv2_2, 'pool2')
# 40*30 256
conv3_1 = conv_layer(pool2, "conv3_1")
conv3_2 = conv_layer(conv3_1, "conv3_2")
conv3_3 = conv_layer(conv3_2, "conv3_3")
conv3_4 = conv_layer(conv3_3, "conv3_4")
pool3 = max_pool(conv3_4, 'pool3')
# 20*15 512
conv4_1 = conv_layer(pool3, "conv4_1")
conv4_2 = conv_layer(conv4_1, "conv4_2")
conv4_3 = conv_layer(conv4_2, "conv4_3")
conv4_4 = conv_layer(conv4_3, "conv4_4")
pool4 = max_pool(conv4_4, 'pool4')
# 10*7 512
conv5_1 = conv_layer(pool4, "conv5_1")
conv5_2 = conv_layer(conv5_1, "conv5_2")
conv5_3 = conv_layer(conv5_2, "conv5_3")
conv5_4 = conv_layer(conv5_3, "conv5_4")

# deconvolution
# 10*7 512
deconv6 = deconv_layer(conv5_4, 512, 64, 1)
conv6 = conv_layer2(deconv6, 64, 64, "conv6")
# 20*14
deconv7 = deconv_layer(conv6, 64, 8, 0)
conv7 = conv_layer2(deconv7, 8, 8, "conv7")
# 40*28
deconv8 = deconv_layer(conv7, 8, 1, 0)
conv8 = conv_layer2(deconv8, 1, 1, "conv8")
# 80*56
y_ = tf.reshape(conv8, [-1, Out_Width*Out_Height])




# loss function
loss = tf.reduce_sum (tf.square(y_-y_reshape))
train = tf.train.AdamOptimizer(lr).minimize(loss)

data = dataload.load_data(is_training)

saver = tf.train.Saver()

sess = tf.Session()


if is_training:
    sess.run(tf.initialize_all_variables())
    # saver.restore(sess, "weights/weights73800.ckpt")
else:
    saver.restore(sess,"weights/no_aug/weights.ckpt")

threshold_125 = 1.25
threshold_125_2 = 1.25 ** 2
threshold_125_3 = 1.25 ** 3
delta_125 = 0.0
delta_125_2 = 0.0
delta_125_3 = 0.0
abs_relative = 0.0
sqr_relative = 0.0
rmse_linear = 0.0
rmse_log = 0.0
rmse_scale_inv = 0.0

def count_pixels(depth, result):
    global threshold_125
    global threshold_125_2
    global threshold_125_3
    global delta_125
    global delta_125_2
    global delta_125_3
    im1 = depth/result
    im2 = result/depth
    im = np.maximum(im1, im2)
    delta_125 += np.sum(im < threshold_125)/(80*60)
    delta_125_2 += np.sum(im < threshold_125_2)/(80*60)
    delta_125_3 += np.sum(im < threshold_125_3)/(80*60)
    print("delta_125: %f"%delta_125)
    print("delta_125_2: %f"%delta_125_2)
    print("delta_125_3: %f"%delta_125_3)

for step in range(int(dataload.get_num()/BATCH_SIZE + 1)):
    images, depths = dataload.get_batch(data, BATCH_SIZE)
    # print(depths[0])
    # print(sess.run(conv8, feed_dict={x:images, y:depths}))
    # print("image:", images[8])
    if is_training:
        if step % 1 == 0:
            loss_value = sess.run(loss, feed_dict={x:images, y:depths})
            print(step, loss_value)

        if step % 900 == 0 and step != 0:
            y_value = sess.run(y_, feed_dict={x: images})
            saver.save(sess, "weights/weights"+str(step)+".ckpt")
            # print("conv5_4", np.resize(y_value, [BATCH_SIZE, 112, 112])[0])

        sess.run(train, feed_dict={x: images, y: depths})
    else:
        result = sess.run(conv8, feed_dict={x:images})
        result = result.reshape([BATCH_SIZE, Out_Width, Out_Height])

        N = 80 * 60
        for idx in range(BATCH_SIZE):
            num = idx + BATCH_SIZE * step
            print("-------------%d----------------"%num)
            mask = result[idx] < 1e-3
            result[idx][mask] = 1e-3
            mask = depths[idx] < 1e-3
            depths[idx][mask] = 1
            result[idx][mask] = 1
            count_pixels(depths[idx], result[idx])
            abs_relative += np.sum(abs(depths[idx] - result[idx])/depths[idx])/N
            sqr_relative += np.sum((depths[idx] - result[idx]) ** 2 / depths[idx]) / N
            rmse_linear += np.sum((depths[idx] - result[idx]) ** 2) / N

            d = (np.log(depths[idx]) - np.log(result[idx]))
            rmse_log += np.sum( d ** 2) / N
            rmse_scale_inv += (np.sum(d ** 2) / N - np.sum(d) ** 2 / (N ** 2))
            print("abs_relative: %f"%abs_relative)
            print("sqr_relative: %f"%sqr_relative)
            print("rmse_linear: %f"%rmse_linear)
            print("rmse_log: %f"%rmse_log)
            print("rmse_scale_inv: %f"%rmse_scale_inv)

            if num == 639:
                print("delta_125: %f"%(delta_125/640))
                print("delta_125_2: %f"%(delta_125_2/640))
                print("delta_125_3: %f"%(delta_125_3/640))
                print("abs_relative: %f"%(abs_relative/640))
                print("sqr_relative: %f"%(sqr_relative/640))
                print("rmse_linear: %f"%np.sqrt(rmse_linear/640))
                print("rmse_log: %f"%np.sqrt(rmse_log/640))
                print("rmse_scale_inv: %f"%np.sqrt(rmse_scale_inv/640))
                break



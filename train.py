import tensorflow as tf
import numpy as np
import input_data
from scipy import misc
from matplotlib import pyplot as plt

def initWeight(shape):
    weights = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(weights)

def initBias(shape):
    bias = tf.constant(0.1,shape=shape)
    return tf.Variable(bias)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def maxPool2d(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

sess = tf.InteractiveSession()



# NOW FOR THE GRAPH BUILDING
x = tf.placeholder("float", shape=[None, 1024])
y_ = tf.placeholder("float", shape=[None, 4])

# turn the pixels into the a matrix
xImage = tf.reshape(x,[-1,32,32,1])
# xImage = x;

# conv layer 1
wConv1 = initWeight([5,5,1,32])
bConv1 = initBias([32])
# turns to 16x16 b/c pooling
hConv1 = tf.nn.relu(conv2d(xImage,wConv1) + bConv1)
hPool1 = maxPool2d(hConv1)

# conv layer 2
wConv2 = initWeight([5,5,32,64])
bConv2 = initBias([64])
# turns to 8x8 b/c pooling
hConv2 = tf.nn.relu(conv2d(hPool1,wConv2) + bConv2)
hPool2 = maxPool2d(hConv2)

# fully connected layer
W_fc1 = initWeight([8 * 8 * 64, 1024])
b_fc1 = initBias([1024])

# resize the 7x7x64 into a 1-D array so we can matmul it.
h_pool2_flat = tf.reshape(hPool2, [-1, 8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout for the FC layer.
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# weights to turn to softmax classify
W_fc2 = initWeight([1024, 4])
b_fc2 = initBias([4])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv  + 1e-9))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


sess.run(tf.initialize_all_variables())

si = 2;
sl = ["j","k"]

batch = np.zeros((si*6,1024))
t = 0
for i in range(si):
    for x in range(6):
        t += 1
        loc = sl[i]+""+x+".jpg"
        print loc
        # img = misc.imread(sl[i]+""+x+".jpg")

batch[0] = misc.imread('j1.jpg').flatten()
batch[1] = misc.imread('liliaSmall.jpg').flatten()
batch[2] = misc.imread('michaelSmall.jpg').flatten()
batch[3] = misc.imread('kevinSmall.jpg').flatten()
batch = batch/225.0
# batch[1] = np.expand_dims(misc.imread('liliaSmall.jpg'),axis=2)
# batch[2] = np.expand_dims(misc.imread('michaelSmall.jpg'),axis=2)
# batch[3] = np.expand_dims(misc.imread('kevinSmall.jpg'),axis=2)

labels = np.zeros((4,4))
labels[0] = np.array([1,0,0,0])
labels[1] = np.array([0,1,0,0])
labels[2] = np.array([0,0,1,0])
labels[3] = np.array([1,0,0,0])
print labels

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# mnistbatch = mnist.train.next_batch(5)
# batch = mnistbatch[0]
# labels = mnistbatch[1]
#
# print batch.shape
print batch[0][:]
# print labels.shape

# batch = mnist.train.next_batch(3)
for i in range(20000):
    # batch = mnist.train.next_batch(50)
    # print batch[0]
    # print type(batch[0])
    if i%10 == 0:
        print "hi"
        train_accuracy = accuracy.eval(feed_dict={x:batch, y_: labels, keep_prob: 1.0})
        print "step %d, training accuracy %g"%(i, train_accuracy)
        result = y_conv.eval(feed_dict={x: batch, y_: labels, keep_prob: 1.0})
        # for k in range(4):
            # print "lmao %d %s" % (k, np.array_str(result[k]))

    train_step.run(feed_dict={x: batch, y_: labels, keep_prob: 0.5})

  # print i

# print("test accuracy %g"%accuracy.eval(feed_dict={
#     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

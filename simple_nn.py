import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import ljq_utils
import utils
import os
import numpy as np

def inference(x, weights, biases, keep_prob):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.tanh(layer_1)
    layer_1 = tf.add(layer_1, tf.ones_like(layer_1))

    layer_2 = tf.matmul(layer_1, weights['h2']) + biases['b2']
    layer_2 = tf.nn.tanh(layer_2)
    layer_2 = tf.add(layer_2, tf.ones_like(layer_2))

    layer_3 = tf.matmul(layer_2, weights['h3']) + biases['b3']
    layer_3 = tf.nn.tanh(layer_3)
    layer_3 = tf.add(layer_3, tf.ones_like(layer_3))

    layer_4 = tf.matmul(layer_3, weights['h4']) + biases['b4']
    layer_4 = tf.nn.tanh(layer_4)
    layer_4 = tf.add(layer_4, tf.ones_like(layer_4))

    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    out_layer = tf.nn.dropout(out_layer, keep_prob)
    out_layer = tf.nn.relu(out_layer)
    return out_layer

# sess = tf.InteractiveSession()
#
x = tf.placeholder(tf.float32,[1, 1000])
y = tf.Variable(tf.zeros([1,250]))
# w = tf.Variable(tf.zeros([1000,250]))
# b = tf.Variable(tf.zeros([250,1]))
#
# y_pre = tf.nn.relu(tf.matmul(x,w) + b)
#
# y_true = tf.placeholder(tf.float32,[250,1])
# loss = tf.reduce_mean(tf.square(y_true-y_pre, name='loss'))
#
# train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# tf.global_variables_initializer().run()

n_hidden_1 = 875
n_hidden_2 = 750
n_hidden_3 = 625
n_hidden_4 = 500
n_input = 1000
n_classes = 250

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

keep_prob = tf.placeholder("float")
predictions = inference(x, weights, biases, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
utils.safe_mkdir('checkpoints')
utils.safe_mkdir('checkpoints/convnet_mnist')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_mnist/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    step=0
    for i in range(10000):
        total_loss=0

        for idx in xrange(1, 24):
            data_train=ljq_utils.load_data('./data/unkown/train/train_{}.txt'.format(str(idx)))
            data_label = ljq_utils.load_label('./data/unkown/train/label_{}.txt'.format(str(idx)))
            # _, l = sess.run([train, loss],feed_dict={x:data_train,y_true:data_label})
            _, l = sess.run([optimizer, cost],feed_dict={x:data_train,y:data_label,keep_prob:0.8})
            total_loss += l
            step+=1
            print('Epoch {0}: {1}'.format(idx, l))
            if np.mod(step,500)==1:
                saver.save(sess, 'checkpoints/unkown/unkown', step)


# correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(y_true,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# print(accuracy.eval({x: mnist.test.images,y_true: mnist.test.labels}))


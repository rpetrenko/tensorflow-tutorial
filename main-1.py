import tensorflow as tf
import time


def generate_linear_points(w, b, n):
    x = list(range(1, n + 1))
    f = lambda x: x * w + b
    return x, [f(a) for a in x]

# Creates a graph.
# a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
# b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
# Runs the op.
# config = tf.ConfigProto(log_device_placement=True)
# sess = tf.Session(config=config)
# print(sess.run(c))

sess = tf.Session()
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # tf.float32 is default
print(node1, node2)
print(sess.run([node1, node2]))

# adding two node constants
node3 = tf.add(node1, node2)
print('node3 ', node3)
print("sess.run(node3) ", sess.run(node3))

# create parameterized graph
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
add_two_nodes = a + b

print(sess.run(add_two_nodes, {a: 3, b: 4.5})) # keys should match node names

# create linear model
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# initialize all Variables
init = tf.global_variables_initializer()
sess.run(init)

# since x is a placeholder, we can compute simultaneously multiple values
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

# compute loss function
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
x_train, y_train = generate_linear_points(-1, 1, 4)
xy_pairs = {x: x_train, y: y_train}

print(sess.run(loss, xy_pairs))

# let's check the actual solution y = -x + 1
trueW = tf.assign(W, [-1.])
trueb = tf.assign(b, [1.])
sess.run([trueW, trueb])
print(sess.run(loss, xy_pairs))

# let's train our model now
# the simplest optimizer is gradient descent
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) # reset values back to incorrect 0.3 and -.3
start_time = time.time()
for i in range(10**5):
    sess.run(train, xy_pairs)
print("Timing ", time.time() - start_time)
# 30 secs for 4 points for 10^5 iters on Titan X, 3072 cores

W_out, b_out, loss_out = sess.run([W, b, loss], xy_pairs)
print("W: %s b: %s loss: %s"%(W_out, b_out, loss_out))
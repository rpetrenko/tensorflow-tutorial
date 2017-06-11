# manual device placement
"""
Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1080, pci bus id: 0000:02:00.0)
Creating TensorFlow device (/gpu:1) -> (device: 1, name: GeForce GTX TITAN X, pci bus id: 0000:01:00.0)
"""
import tensorflow as tf
import time

N = 30000
M = 2000
print("generating array")
gen_arr = [float(x) for x in range(N * M)]
dev_str = '/gpu:1'
with tf.device(dev_str):
    a = tf.constant(gen_arr, shape=[N, M], name='a')
    b = tf.constant(gen_arr, shape=[M, N], name='b')
    c = tf.matmul(a, b)

print("start running")
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
st = time.time()
for i in range(1000):
    sess.run(c)
print("Timing", time.time() - st)
# print(sess.run(c))

# /gpu:0 Timing 1209.202615737915
# /gpu:1 Timing 1265.0938534736633

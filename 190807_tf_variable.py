import tensorflow as tf

W1 = tf.Variable(tf.random_normal([1]))
b1 = tf.Variable(tf.random_normal([1]))

W2 = tf.Variable(tf.random_normal([1, 2]))
b2 = tf.Variable(tf.random_normal([1, 2]))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(3):
    W1 = W1 - step
    b1 = b1 - step

    W2 = W2 - step
    b2 = b2 - step

    print("step:", step, "W1:", sess.run(W1), "b1:", sess.run(b1))
    print("step:", step, "W2:", sess.run(W2), "b2:", sess.run(b2))
    
import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = a + b

sess = tf.Session()

print(sess.run(c, feed_dict={a:1., b:3.}))
print(sess.run(c, feed_dict={a:[1., 2.], b:[3., 4.]}))

d = 100 * c

print(sess.run(d, feed_dict={a:1., b:3.,}))
print(sess.run(d, feed_dict={a:[1., 2.], b:[3., 4.]}))
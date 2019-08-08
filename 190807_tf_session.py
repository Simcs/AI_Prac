import tensorflow as tf

a = tf.constant(1.0, name='a')
b = tf.constant(2.0, name='b')
c = tf.constant([[1., 2.], [3., 4.]])

print(a)
print(a + b)
print(c)

sess = tf.Session()
print(sess)
print(sess.run([a, b]))
print(sess.run(c))
print(sess.run([a + b]))
print(sess.run(c + 1.0))

sess.close()
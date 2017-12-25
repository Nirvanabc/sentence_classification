import tensorflow as tf

sess = tf.Session()

node1 = tf.constant(3.0)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2, name='add')
sess.run(tf.global_variables_initializer())
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("output", sess.graph)

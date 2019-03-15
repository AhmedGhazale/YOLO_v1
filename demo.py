import tensorflow as tf
from yolo_v1_net import YOLO
from tf_records import read_dataset

yolo=YOLO()

dataset=read_dataset("train.tfrecords",batch_size=32,repeat=100)

iter = dataset.make_one_shot_iterator()
X_batch,Y_batch = iter.get_next()

iter = 10000
opt =tf.train.AdamOptimizer(yolo.learning_rate).minimize(yolo.total_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range (iter):
        batch_x,batch_y = sess.run(X_batch,Y_batch)
        _, loss = sess.run([opt ,yolo.total_loss ], feed_dict={yolo.images: batch_x, yolo.labels: batch_y})
        print(loss)






